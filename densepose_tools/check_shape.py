from torchvision import transforms
import os
import torch
import torch.nn.functional as F
import numpy as np
tensor_transforms = transforms.Compose(
    [
        transforms.Normalize([0.5], [0.5]),
    ]
)
imSize = (512, 640)
h, w = imSize[1], imSize[0]
# dp_path = '/home/nfs/jsh/DreamPose/demo/sample/poses/demo/sample/poses/frame_50_densepose.npy'
dp_path = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose/00006_00.jpg.npy'
from IPython import embed; embed()
print(np.load(dp_path))
# (2, 720, 576)
print(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0).shape)
# torch.Size([1, 2, 720, 576])
dp_i = F.interpolate(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0), (h, w), mode='bilinear').squeeze(0)
print(dp_i.shape)
# torch.Size([2, 640, 512])
poses = []
trans_dp_i=tensor_transforms(dp_i)
print(trans_dp_i.shape)
# torch.Size([2, 640, 512])
poses.append(trans_dp_i)
input_pose = torch.cat(poses, 0).unsqueeze(0)
print(input_pose.shape)
# torch.Size([1, 2, 640, 512])