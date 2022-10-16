import os
import json
import random
from time import time
from turtle import dot

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from pytorch_fid.fid_score import calculate_fid_given_paths as fid


class Generator(nn.Module):
    def __init__(self, z_dim, channels):
        super(Generator, self).__init__()
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.extactor = nn.Sequential(
            nn.Conv2d(ndf * 8, 100, 4, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        backbone = self.main(input)
        c = self.classifier(backbone)
        f = self.extactor(backbone)
        return c, f


class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
    
    def forward(self, x, isDisc):
        dot_product = 0
        hb = int(len(x) / 2)
        for i in range(hb):
            dot_product += torch.dot(x[i].view(-1), x[i+hb].view(-1))
        loss_base = torch.abs(dot_product / hb / 2)

        if isDisc:
            return 100 - loss_base
        else:
            return loss_base


class DCGAN_LFM:
    def __init__(self) -> None:
        torch.set_flush_denormal(True)


    def config(self, conf):
        self.config_dict = conf

        # hyper parameters
        self.lrg = conf["lrg"]                 # learning rate for the Generator
        self.lrd = conf["lrd"]                 # learning rate for the Discriminator
        self.z_dim = conf["z_dim"]             # latent space size
        self.batch_size = conf["batch_size"]   # batch size
        
        # training related settings
        self.seed = conf["seed"]
        self.max_iter = conf["max_iter"]

        self.lamG = conf["lamG"]
        self.lamD = conf["lamD"]

        # dataset settings
        self.channels = conf["channels"]
        self.data_dir = conf["data_dir"]
        self.fid_gt = conf["fid_gt"]           # fid ground truth, as .npz file

        # computer settings
        self.device = conf["device"]
        self.workers = conf["workers"]
        self.save_iter = conf["save_iter"]
        self.result_dir = conf["result_dir"]

        self.grid_dir = self.result_dir + "grids\\"
        self.pic_dir = self.result_dir + "pics\\"
        self.model_dir = self.result_dir + "models\\"

        os.makedirs(self.result_dir + "grids\\", mode=0o777, exist_ok=True)
        os.makedirs(self.result_dir + "pics\\", mode=0o777, exist_ok=True)
        os.makedirs(self.result_dir + "models\\", mode=0o777, exist_ok=True)

        random.seed(self.seed)
        torch.manual_seed(self.seed)
    

    def load_data(self):
        if self.channels == 3:
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif self.channels == 1:
            norm = transforms.Normalize((0.5,), (0.5,))
        else:
            print("Image channel should be 1 or 3.")
            exit()

        dataset = dset.ImageFolder(
            root=self.data_dir,
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                norm,
            ])
        )
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
    

    def prepare_training(self):

        self.generator = Generator(self.z_dim, self.channels).to(self.device)
        self.discriminator = Discriminator(self.channels).to(self.device)

        # initialize weight
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        # init optimizer and loss function
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lrd, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.lrg, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.feature_c = FeatureLoss()

        with open(self.result_dir + "config.json", "w") as current_config:
            json.dump(self.config_dict, current_config)

        # log file
        with open(self.result_dir + "log.csv", "w") as log:
            log.write("index, lossD, lossG, acc_real, acc_fakem, time, fid\n")

        self.fixed_noise = torch.randn((128, self.z_dim, 1, 1)).to(self.device)
    

    def save_models(self, name):
        torch.save(self.generator, self.model_dir + name + "G.pth")
        torch.save(self.discriminator, self.model_dir + name + "D.pth")
    

    def save_pics(self, name, images):
        with torch.no_grad():
            fake = self.generator(self.fixed_noise).detach().cpu()
        for i, img in enumerate(fake):
            os.makedirs(self.pic_dir + name, mode=0o777, exist_ok=True)
            torchvision.utils.save_image(img, self.pic_dir + name + "\\{}.png".format(i), normalize=True)
        grid = torchvision.utils.make_grid(fake, nrow=16, normalize=True)
        torchvision.utils.save_image(grid, self.grid_dir + "{}.png".format(name))


    def weights_init(m, params):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    

    def generate_noise(self):
        batch_size = self.batch_size
        z_dim = self.z_dim

        noise = torch.zeros((batch_size, z_dim, 1, 1))
        count = 0
        while count < batch_size / 2:
            n1 = torch.randn((z_dim,)) 
            n2 = torch.randn((z_dim,))
            dr = torch.dot(n1[0:z_dim-1], n2[0:z_dim-1])
            n2[z_dim-1] = -dr / n1[z_dim-1]
            if abs(n2[z_dim-1]) > 1.5:
                continue
            else:
                noise[count] = n1.view((z_dim, 1, 1))
                noise[int(count + batch_size / 2)] = n2.view((z_dim, 1, 1))
                count += 1
        
        return noise
    

    def train(self):
        iter = 0
        start = time()
        self.load_data()
        self.prepare_training()
        hb = int(self.batch_size / 2)


        while iter < self.max_iter:
            for i, (real, _) in enumerate(self.dataloader):
                
                noise = self.generate_noise().to(self.device)

                # the -1 used here instead of (batch_size, image_dim)
                # is because the last batch might just have few images left
                real = real.to(self.device)  # reshape
                fake = self.generator(noise)

                ### train discriminator: max lod(D(real)) + log(1 - D(G(z)))

                disc_real, _ = self.discriminator(real)
                disc_real = disc_real.view(-1)
                disc_fake, disc_f = self.discriminator(fake)
                disc_fake = disc_fake.view(-1)
                
                lossD_real = self.criterion(disc_real, torch.ones_like(disc_real))
                lossD_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD_class = (lossD_real + lossD_fake) / 2
                
                lossD_feature = 100 - self.feature_c(disc_f, True)

                lossD = (lossD_class + lossD_feature) / 2
               

                self.discriminator.zero_grad()
                lossD.backward(retain_graph=True)
                self.optimizerD.step()


                ### Train Generator max log(D(G(z)))

                output, output_f = self.discriminator(fake)
                output = output.view(-1)
                lossG_class = self.criterion(output, torch.ones_like(output))

                lossG_feature = self.feature_c(output_f, False)

                lossG = (lossG_class + lossG_feature) / 2

                self.generator.zero_grad()
                lossG.backward()
                self.optimizerG.step()


                print("{}".format(iter), end="\r")
                if iter % self.save_iter == 0:
                    with torch.no_grad():
                        # fake = self.generator(self.fixed_noise).detach().cpu()
                        fake = self.generator(self.fixed_noise)
                    self.save_pics(str(iter), fake)
                    fid_score = fid (
                            paths=[
                                self.pic_dir + "{}\\".format(iter), 
                                self.fid_gt
                                ],
                            batch_size=128,
                            device="cuda",
                            dims=2048,
                            num_workers=0
                        )
                    acc_real = disc_real.mean().item()
                    acc_fake = 1 - disc_fake.mean().item()
                    with open(self.result_dir + "log.csv", "a") as log:
                        log.write("{}, {:.3f}, {:.3f}, {:.2f}, {:.2f}, {:.0f}, {:.2f}\n".format(iter, lossD, lossG, acc_real, acc_fake, time()-start, fid_score))
                    print("\r{}, lossD: {:.3f}, lossG: {:.3f}, acc_real: {:.2f}, acc_fake: {:.2f}, time: {:.0f}, FID: {:.2f}".format(iter, lossD, lossG, acc_real, acc_fake, time()-start, fid_score))
                    self.save_models("{}".format(iter))
                
                torch.clear_autocast_cache()
                iter += 1
                if iter >= self.max_iter:
                    break



if __name__ == '__main__':

    config = {
        "lrg": 2e-4,
        "lrd": 2e-4,
        "z_dim": 100,
        "batch_size": 128,

        "seed": 0,
        "max_iter": 300000,

        "lamG": 0.5,
        "lamD": 0.5,

        "channels": 3,
        "data_dir": "E:\\Caldron\\GANBOX\\dataset\\celeba_cropped\\",
        "fid_gt": "E:\\Caldron\\GANBOX\\result\\pretrained_FID\\fid_stats_celeba.npz",

        "device": "cuda",
        "workers": 0,
        "save_iter": 500,
        "result_dir": "result\\DCGAN\\",
    }
    torch.set_flush_denormal(True)
    gan = DCGAN_LFM()
    gan.config(config)
    gan.train()