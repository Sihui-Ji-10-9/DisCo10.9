# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")
'''
from huggingface_hub import snapshot_download
snapshot_download(repo_id="facebook/dinov2-large", cache_dir='huggingface/hub')
'''

from torchvision import transforms
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
tensor_transforms = transforms.Compose(
    [
        transforms.Normalize([0.5], [0.5]),
    ]
)
imSize = (512, 640)
h, w = imSize[1], imSize[0]
dp_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose/00006_00.jpg.npy'
dp_path1 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose/00008_00.jpg.npy'

# dp_path = '/home/nfs/jsh/HierarchicalProbabilistic3DHuman/output/my2/range/iuv_0.npy'
dp_i_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose_i/00006_00.jpg.npy'
dp_i_path1 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose_i/00008_00.jpg.npy'
# data10 = np.load(dp_path1)[ 0, :, :].round(decimals=6)
# data11 = np.load(dp_path1)[ 1, :, :].round(decimals=6)
data1_uv = np.load(dp_path1)
data1_i = np.load(dp_i_path1)

data1 = np.concatenate([data1_uv,data1_i],axis=0)
print(data1.shape)

# data00 = np.load(dp_path0)[0, :, :].round(decimals=6)
# data01 = np.load(dp_path0)[1, :, :].round(decimals=6)
data0_uv = np.load(dp_path0)
data0_i = np.load(dp_i_path0)
# print(data0_i)
data0 = np.concatenate([data0_uv,data0_i],axis=0)
print(data0.shape)

c = np.empty_like(data0, dtype=list)
# print(c)
for k in range(3):
    item=np.where((data0!=0)&(data0==data1[2, 300, 300]))

    if item!=None:
        coordinates = list(zip(item[0], item[1],item[2]))
        c[k,i,j]=coordinates

for k in range(3):
    print('--',k)
    for i in range(300,301):
        print('--',i)
        for j in range(300,301):
            print('--',j)
            print(c[k,i,j])
