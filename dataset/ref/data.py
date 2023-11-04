import os
import os.path as osp
from config import *
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from dataset.tsv_cond_dataset import TsvCondImgCompositeDataset
# from dataset.sparse_vis_optical_flow import *
from typing import Optional
import random
import pdb


#将前景按照mask，切出来

class BaseDataset(TsvCondImgCompositeDataset):
    def __init__(self, args, yaml_file, split='train', preprocesser=None):
        
        self.split = split
        self.img_size = getattr(args, 'img_full_size', args.img_size)
        self.basic_root_dir = BasicArgs.root_dir
        self.max_video_len = args.max_video_len
        assert self.max_video_len == 1
        self.is_composite = False
        super().__init__(args, yaml_file, split=split,size_frame=args.max_video_len, tokzr=None)
        self.img_ratio = (1., 1.) if not hasattr(self.args, 'img_ratio') or self.args.img_ratio is None else self.args.img_ratio
        self.img_scale = (1., 1.) if not split=='train' else getattr(self.args, 'img_scale', (0.9, 1.0)) # val set should keep scale=1.0 to avoid the random crop
        print(f'Current Data: {split}; Use image scale: {self.img_scale}; Use image ratio: {self.img_ratio}')
        
        self.train_lst = args.train_lst
        self.val_lst = args.val_lst
        self.data_lst = []
        
        if split == 'train':
            for cur_f in self.train_lst:
                self.data_lst += open(cur_f).readlines()
        else:
            for cur_f in self.val_lst:
                self.data_lst += open(cur_f).readlines()
            tt = len(self.data_lst)//500
            self.data_lst = self.data_lst[::tt]

        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(self.img_size,scale=self.img_scale, ratio=self.img_ratio,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(self.img_size,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_msk = transforms.Compose([
            # transforms.RandomResizedCrop(self.img_size,scale=self.img_scale, ratio=self.img_ratio,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(self.img_size,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.transform_dino = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            # transforms.RandomResizedCrop((224, 224),scale=self.img_scale, ratio=self.img_ratio,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize(self.img_size,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225]),
        ])
        self.transform_clip_msk = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            # transforms.RandomResizedCrop((224, 224),scale=self.img_scale, ratio=self.img_ratio,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize((224, 224),interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        self.length = len(self.data_lst)

    def augmentation(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)
    
    def get_file_path(self, cur_img_dir, source_img_key, tgt_img_key):
        if 'UBC' in cur_img_dir:
            img_suf = '.png'
        elif 'In-shop_Clothes_Retrieval_Benchmark' in cur_img_dir:
            img_suf = '.jpg'
        else:
            raise NotImplementedError()
        source_img_path = osp.join(cur_img_dir, source_img_key+img_suf)#数据集不一样，图片格式不同
        # source_pose_path = osp.join(cur_img_dir.replace('/img/', '/openpose_img/'), source_img_key+'.png') #固定存为png
        source_mask_path = osp.join(cur_img_dir.replace('/img/', '/mask/'), source_img_key+'.png') #固定存为png

        # print(cur_img_dir, tgt_img_key, img_suf)
        tgt_img_path = osp.join(cur_img_dir, tgt_img_key+img_suf)#数据集不一样，图片格式不同
        tgt_pose_path = osp.join(cur_img_dir.replace('/img/', '/openpose_img/'), tgt_img_key+'.png') #固定存为png
        
        return source_img_path, source_mask_path, tgt_img_path, tgt_pose_path
    
    def get_metadata(self, idx):
        # pdb.set_trace()
        idx = idx%self.length

        data_info = self.data_lst[idx].rstrip().split(',')
        cur_img_dir = data_info[0]
        cur_vid_key = cur_img_dir.split('/')[-1]
        img_key_lst = data_info[1:]
        source_img_key = img_key_lst[0] # if self.is_inference else random.SystemRandom().sample(img_key_lst)
        tgt_img_key = random.SystemRandom().sample(img_key_lst[1:], 1)[0] if self.split == 'train' else img_key_lst[1:][len(img_key_lst[1:])//2]
        source_img_path, source_mask_path, tgt_img_path, tgt_pose_path = self.get_file_path(cur_img_dir, source_img_key, tgt_img_key)

        img_key = cur_vid_key
        pose_img = Image.open(tgt_pose_path)
        gt_img = Image.open(tgt_img_path)
        input_img = Image.open(source_img_path)
        input_msk = Image.open(source_mask_path)

        #获取更加紧凑的前景，先mask 再根据最小外接矩形crop
        msk_np = cv2.imread(source_mask_path)//255
        input_img_np = np.array(input_img)
        y_lst, x_lst = np.nonzero(msk_np[:,:,0])
        x_min = x_lst.min()
        x_max = x_lst.max()
        y_min = y_lst.min()
        y_max = y_lst.max()
        croped_body = Image.fromarray(input_img_np * msk_np).crop((x_min,y_min,x_max,y_max))
        # croped_body.save(f'ttt/{idx}.jpg')

        # preparing outputs
        meta_data = {}
        meta_data['img_key'] = f"{cur_vid_key}_{source_img_key}_{tgt_img_key}"
        meta_data['pose_img'] = pose_img
        meta_data['img'] = gt_img
        meta_data['reference_img'] = input_img
        meta_data['reference_msk'] = input_msk
        meta_data['croped_body'] = croped_body
        return meta_data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # pdb.set_trace()
        raw_data = self.get_metadata(idx)
        #dylee
        img_key = raw_data['img_key'] 
        tgt_img = raw_data['img']   #gt
        skeleton_img = raw_data['pose_img'] #flow
        reference_img = raw_data['reference_img']   #input img
        reference_msk = raw_data['reference_msk']
        croped_body = raw_data['croped_body']
        w,h = tgt_img.size #PIL.Image
        
        state = torch.get_rng_state()
        
        tgt_img = self.augmentation(tgt_img, self.transform, state)
        skeleton_img = self.augmentation(skeleton_img, self.transform, state)
        background_msk = 1 - self.augmentation(reference_msk, self.transform_msk, state)
        # foreround_msk = self.augmentation(reference_msk, self.transform_clip_msk, state)
        ori_img = self.augmentation(reference_img, self.transform, state)
        reference_img_contronet = ori_img * background_msk
        # reference_img = self.augmentation(reference_img, self.transform_clip, state) * foreround_msk
        reference_img = self.augmentation(croped_body, self.transform_dino, state)      #使用比较紧凑的前景信息
        # reference_img = transforms.Resize(self.img_size,interpolation=transforms.InterpolationMode.BILINEAR)(croped_body)

        #                                                       #gt                 #                           # .. of input img
        # outputs = {'img_key':img_key, 'input_text': caption, 'label_imgs': img, 'cond_imgs': skeleton_img, 'ref_cond_imgs': ref_skeleton_img, 'reference_img_ori': reference_img_ori, 'reference_img': reference_img_new_fg, 'reference_img_controlnet':reference_img_ori, 'reference_img_vae':reference_img_vae}
        outputs = {'img_key':img_key, 'ori_img': ori_img, 'label_imgs': tgt_img, 'cond_imgs': skeleton_img, 'reference_img': reference_img, 'reference_img_controlnet':reference_img_contronet} #, 'reference_img_vae':reference_img_vae}

        return outputs
