import os
import shutil

target = "E:\\Caldron\\LFM\\result\\"
f = os.walk(target)

for file in f:
    if 'models' in file[0]:
        shutil.rmtree(file[0])