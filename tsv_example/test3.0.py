import os
import numpy as np


# 定义均值和标准差
mean = np.array([0.5])
std = np.array([0.5])

folder_path_uv = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_big'
folder_path_i = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_i_big'
file_paths_uv = [os.path.join(folder_path_uv, file) for file in os.listdir(folder_path_uv) if file.endswith('.npy')]
file_paths_i = [os.path.join(folder_path_i, file) for file in os.listdir(folder_path_i) if file.endswith('.npy')]
file_paths_uv.sort()
file_paths_i.sort()
print(file_paths_uv)
print(file_paths_i)

directory = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_fat_cor_new11.0'
os.makedirs(directory, exist_ok=True)

# dp_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose/00006_00.jpg.npy'
dp_path0 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_big/00069_00 copy 2.jpg.npy'
# dp_i_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose_i/00006_00.jpg.npy'
dp_i_path0 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_i_big/00069_00 copy 2.jpg.npy'

data0_uv = np.load(dp_path0)
data0_i = np.load(dp_i_path0)
data0_uv = (data0_uv - mean) / std
print('data0_uv',data0_uv.shape)
print('data0_i',data0_i.shape)
data0 = np.concatenate([data0_uv,data0_i],axis=0)
cor=[]
# from IPython import embed; embed()
for i in range(len(file_paths_uv)):
    path_uv_i = file_paths_uv[i]
    path_i_i = file_paths_i[i]
    data_uv_i = np.load(path_uv_i)
    data_uv_i = (data_uv_i - mean) / std
    data_i_i = np.load(path_i_i)
    data_i = np.concatenate([data_uv_i,data_i_i],axis=0)
    print(data_i.shape)
    # c = np.empty_like(data_i[0], dtype=list)
    c = np.ones((1024,768,2))*(-100000)
    for i in range(1024):
        for j in range(768):
            # item=np.where(data0[:]==data1[:,i,j])
            if data0[2,i,j]!=0:
                item=np.where((np.abs(data_i[0]-data0[0,i,j])<0.02)&(np.abs(data_i[1]-data0[1,i,j])<0.02)&(np.abs(data_i[2]-data0[2,i,j])<0.5))
                # item=np.where((data_i[0]==data0[0,i,j])&(data_i[1]==data0[1,i,j])&(data_i[2]==data0[2,i,j]))
                # print(item)
                if item[0].size > 0:
                    # coordinates = list(zip(item[0], item[1]))
                    # d[i,j]=coordinates
                    # print('item',item)
                    c[i,j,0]=np.mean(item[0]).round()
                    c[i,j,1]=np.mean(item[1]).round()
                    # c[i,j]=[np.mean(item[0]).round(),np.mean(item[1]).round()]
    save_path = path_uv_i.replace('densepose_big', 'densepose_fat_cor_new11.0')
    np.save(save_path, c,allow_pickle=True)
    cor.append(c)
cor = np.array(cor)
np.save('/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/cor11.0.npy', cor,allow_pickle=True)
print(cor.shape)
# 10 1024 768 2
# 6-16个点

# cc = np.load(save_path,allow_pickle=True)
# # print(c)
# cnt=0
# for i in range(100,200):
#     for j in range(350,768):
#             if cc[i,j,0]!=-1:
#                 # print(c[i,j])
#                 print((cc[i,j,0],cc[i,j,1]),(i,j))
#                 cnt+=1
# print(cc.shape)
# print(cnt)

# cc = np.load('/HOME/HOME/jisihui/VITON-hd-resized/try2.0/densepose_fat_cor/00069_00 copy 2.jpg.npy',allow_pickle=True)
# cnt=0
# for i in range(100,200):
#     for j in range(350,768):
#             if cc[i,j]!=None:
#                 # print(c[i,j])
#                 print(cc[i,j],i,j)
#                 cnt+=1
