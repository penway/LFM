from DCGAN import DCGAN
from LFMz import DCGAN_LFM

config = {
    "lrg": 2e-4,
    "lrd": 2e-4,
    
    "z_dim": 100,
    "batch_size": 128,

    "seed": 37,
    "max_iter": 1000001,

    "lamG": 0.2,
    "lamD": 0.2,

    "channels": 3,
    "data_dir": "D:\\regis\\Documents\\celeba_0.1\\",
    # "data_dir": "D:\\regis\\Documents\\Datasets\\met\\",
    "fid_gt": "E:\\Caldron\\GANBOX\\result\\pretrained_FID\\celeba_pretrained_FID.npz",

    "device": "cuda",
    "workers": 0,
    "save_iter": 500,
    "result_dir": "result\\LFMzabs_0.1_2\\",
}

lfm = DCGAN_LFM()
lfm.config(config)
lfm.train()

# for p in [0.1, 1]:

#     config["data_dir"] = "D:\\regis\\Documents\\Datasets\\celeba_{}".format(p)

#     config["result_dir"] = "result\\DC_{}\\".format(p)
#     print("\nDC_{}".format(p))
#     dc = DCGAN()
#     dc.config(config)
#     dc.train()


    # config["result_dir"] = "result\\LFMzabs_{}\\".format(p)
    # print("\nLFMzabs_{}".format(p))
    # lfm = DCGAN_LFM()
    # lfm.config(config)
    # lfm.train()
    
