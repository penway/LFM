from DCGAN import DCGAN
from LFM import DCGAN_LFM

config = {
    "lrg": 2e-4,
    "lrd": 2e-4,
    "z_dim": 100,
    "batch_size": 128,

    "seed": 0,
    "max_iter": 30001,

    "lamG": 0.5,
    "lamD": 0.5,

    "channels": 3,
    "portion": 0.001,
    "data_dir": "D:\\regis\\Documents\\Datasets\\celeba_cropped\\",
    "fid_gt": "E:\\Caldron\\GANBOX\\result\\pretrained_FID\\fid_stats_celeba.npz",

    "device": "cuda",
    "workers": 0,
    "save_iter": 500,
    "result_dir": "result\\LFM_0.001\\",
}


for p in [1, 0.5, 0.1, 0.02]:

    config["portion"] = p
    config["result_dir"] = "result\\MNIST\\LFM_{}\\".format(p)
    
    print("\nLFM_{}".format(p))
    lfm = DCGAN_LFM()
    lfm.config(config)
    lfm.train()
    
    config["result_dir"] = "result\\MNIST\\DC_{}\\".format(p)

    print("\nDC_{}".format(p))
    dc = DCGAN()
    dc.config(config)
    dc.train()