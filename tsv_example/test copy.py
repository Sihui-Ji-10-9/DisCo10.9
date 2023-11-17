import os
import numpy as np

folder_path_uv = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_fat'
folder_path_i = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_i_fat'
file_paths_uv = [os.path.join(folder_path_uv, file) for file in os.listdir(folder_path_uv) if file.endswith('.npy')]
file_paths_i = [os.path.join(folder_path_i, file) for file in os.listdir(folder_path_i) if file.endswith('.npy')]
file_paths_uv.sort()
file_paths_i.sort()
print(file_paths_uv)
print(file_paths_i)


# dp_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose/00006_00.jpg.npy'
dp_path0 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_fat/00069_00 copy 2.jpg.npy'
# dp_i_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose_i/00006_00.jpg.npy'
dp_i_path0 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_i_fat/00069_00 copy 2.jpg.npy'

data0_uv = np.load(dp_path0).round(decimals=2)
data0_i = np.load(dp_i_path0)
data0 = np.concatenate([data0_uv,data0_i],axis=0)

# from IPython import embed; embed()
for i in range(1,2):
    path_uv_i = file_paths_uv[i]
    path_i_i = file_paths_i[i]
    data_uv_i = np.load(path_uv_i).round(decimals=2)
    data_i_i = np.load(path_i_i)
    data_i = np.concatenate([data_uv_i,data_i_i],axis=0)
    print(data_i.shape)
    # c = np.empty_like(data_i[0], dtype=list)
    c = np.ones((1024,768,2))*(-1)
    for i in range(100,500):
        for j in range(350,768):
            # item=np.where(data0[:]==data1[:,i,j])
            if data0[0,i,j]!=0:
                item=np.where((data_i[0]==data0[0,i,j])&(data_i[1]==data0[1,i,j])&(data_i[2]==data0[2,i,j]))
                # print(item)
                if item[0].size > 0:
                    # coordinates = list(zip(item[0], item[1]))
                    # d[i,j]=coordinates
                    print('item',item)
                    c[i,j,0]=np.mean(item[0]).round()
                    c[i,j,1]=np.mean(item[1]).round()
                    # c[i,j]=[np.mean(item[0]).round(),np.mean(item[1]).round()]
    save_path = path_uv_i.replace('densepose_fat', 'densepose_fat_cor_base')
    np.save(save_path, c,allow_pickle=True)
    print(save_path)
    


# # dp_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose/00006_00.jpg.npy'
# dp_path0 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_fat/00069_00 copy 2.jpg.npy'
# # dp_path1 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose/00008_00.jpg.npy'
# dp_path1 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_fat/00069_00 copy 9.jpg.npy'
# # dp_i_path1 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose_i/00008_00.jpg.npy'
# dp_i_path1 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_i_fat/00069_00 copy 9.jpg.npy'
# # dp_i_path0 = '/HOME/HOME/jisihui/VITON-hd-resized/try/densepose_i/00006_00.jpg.npy'
# dp_i_path0 = '/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_i_fat/00069_00 copy 2.jpg.npy'

# # data10 = np.load(dp_path1)[ 0, :, :].round(decimals=3)
# # data11 = np.load(dp_path1)[ 1, :, :].round(decimals=3)
# data1_uv = np.load(dp_path1).round(decimals=2)
# data1_i = np.load(dp_i_path1)

# data1 = np.concatenate([data1_uv,data1_i],axis=0)
# # print(data1.shape)
# # 1 1024 

# # data00 = np.load(dp_path0)[0, :, :].round(decimals=3)
# # data01 = np.load(dp_path0)[1, :, :].round(decimals=3)
# data0_uv = np.load(dp_path0).round(decimals=2)
# data0_i = np.load(dp_i_path0)

# data0 = np.concatenate([data0_uv,data0_i],axis=0)
# # print(data0.shape)
# # print(data0[:,214.0,292])
# # # import numpy as np
# # # a=np.array([[[1,2],[3,4]],
# # #             [[5,6],[7,8]],
# # #             [[9,10],[11,12]]])
# # # b=np.array([[[1,1],[1,1]],
# # #             [[5,6],[7,8]],
# # #             [[9,10],[11,12]]])
# # # print(a[2].shape)
# c = np.empty_like(data0[0], dtype=list)
# # d = np.empty_like(data0[0], dtype=list)
# # print(data0[1,30,355])
# for i in range(100,200):
#     for j in range(350,768):
#         # item=np.where(data0[:]==data1[:,i,j])
#         if data1[0,i,j]!=0:
#             item=np.where((data0[0]==data1[0,i,j])&(data0[1]==data1[1,i,j])&(data0[2]==data1[2,i,j]))
#             # print(item)
#             if item[0].size > 0:
#                 # coordinates = list(zip(item[0], item[1]))
#                 # d[i,j]=coordinates
#                 print('item',item)
#                 c[i,j]=[np.mean(item[0]).round(),np.mean(item[1]).round()]
# # 将数组保存为二进制文件
# np.save('/home/nfs/jsh/DisCo/tsv_example/c.npy', c,allow_pickle=True)

# cc = np.load('/home/nfs/jsh/DisCo/tsv_example/c.npy',allow_pickle=True)
# cnt=0
# for i in range(100,500):
#     for j in range(350,768):
#             if cc[i,j]!=None:
#                 # print(c[i,j])
#                 print(cc[i,j],i,j)
#                 cnt+=1


# cc = np.load('/home/nfs/jsh/HOME/VITON-hd-resized/try2.0/densepose_fat_cor_base/00069_00.jpg.npy',allow_pickle=True)
# # print(c)
# cnt=0
# for i in range(100,150):
#     for j in range(350,768):
#             if cc[i,j,0]!=-1:
#                 # print(c[i,j])
#                 print((cc[i,j,0],cc[i,j,1]),(i,j))
#                 cnt+=1
# print(cc.shape)
# print(cnt)
