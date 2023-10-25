# make sure DensePose is in your PYTHONPATH, or use the following line to add it:
# sys.path.append("/your_detectron2_path/detectron2_repo/projects/DensePose/")
import pickle
import torch
import sys
import os
import cv2
import glob
import tqdm
import numpy as np 
sys.path.append("/home/nfs/jsh/detectron2/projects/DensePose")
f = open('/home/nfs/jsh/detectron2/projects/DensePose/dump_viton_hd_side3.pkl', 'rb')
data = torch.load(f)

# img_id, instance_id = 1, 0  # Look at the first image and the first detected instance
# bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
# result = data[img_id]['pred_densepose'][instance_id]
# uv = result.uv
# print(uv.shape)

# Filepath to raw DensePose pickle output
# outpath = '../UBC_Fashion_Dataset/detectron2/projects/DensePose/densepose.pkl'
# Convert pickle data to numpy arrays and save
print('==',len(data))
for i in tqdm.tqdm(range(len(data))):
	dp = data[i]
	# print(dp.keys())
	# dict_keys(['file_name', 'scores', 'pred_boxes_XYXY', 'pred_densepose'])
	path = dp['file_name'] # path to original image
	dp_uv = dp['pred_densepose'][0].uv # uv coordinates
	h, w, c = cv2.imread(path).shape
	_, h_, w_ = dp_uv.shape
	# print(dp_uv.shape)
	# torch.Size([2, 1097, 386])
	(x1, y1, x2, y2) = dp['pred_boxes_XYXY'][0].int().numpy() # location of person
	y2, x2 = y1+h_, x1+w_
	dp_im = np.zeros((2, h, w))
	dp_im[:,y1:y2,x1:x2] = dp_uv.cpu().numpy()
	# savepath1 = path.replace('.png', '_densepose.npy')
	# savepath = savepath1.replace('image', 'densepose')
	savepath1 = path.replace('.jpg', '_densepose.npy')
	savepath = savepath1.replace('Jackets_Vests', 'Jackets_Vests/densepose')
	np.save(savepath, dp_im)

