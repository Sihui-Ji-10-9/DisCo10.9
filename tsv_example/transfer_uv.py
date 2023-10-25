from torchvision import transforms
import os
import torch
import torch.nn.functional as F
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

dp_path0 = '/home/nfs/jsh/HierarchicalProbabilistic3DHuman/output/my/iuv1.npy'
dp_path2 = '/home/nfs/jsh/HierarchicalProbabilistic3DHuman/output/my/iuv1_45.npy'
dp_path1 = '/home/nfs/jsh/HierarchicalProbabilistic3DHuman/output/my/iuv1_22_5.npy'

# data1 = np.load(dp_path0)
data10 = np.load(dp_path0)[ 0, :, :]
data11 = np.load(dp_path0)[ 1, :, :]
data11[data11 != 0] = 1 - data11[data11 != 0]
data1=np.array([data10,data11])
print(data1.shape)
np.save('/home/nfs/jsh/HierarchicalProbabilistic3DHuman/output/my/iuv1.npy', data1)
