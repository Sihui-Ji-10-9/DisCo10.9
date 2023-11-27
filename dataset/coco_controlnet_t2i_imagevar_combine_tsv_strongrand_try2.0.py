from config import *
import torch
import numpy as np
from PIL import Image
from dataset.tsv_cond_dataset import TsvCondImgCompositeDataset


class BaseDataset(TsvCondImgCompositeDataset):
    def __init__(self, args, yaml_file, split='train', preprocesser=None):
        self.img_size = getattr(args, 'img_full_size', args.img_size)
        # self.clip_size = (224,224)
        self.basic_root_dir = BasicArgs.root_dir
        self.max_video_len = args.max_video_len
        assert self.max_video_len == 1
        self.fps = args.fps
        self.dataset = "COCO"
        self.preprocesser = preprocesser
        args.ref_mode = None
        
        super().__init__(
            args, yaml_file, split=split,
            size_frame=args.max_video_len, tokzr=None)
        self.data_dir = args.data_dir

        self.random_square_height = transforms.Lambda(lambda img: transforms.functional.crop(img, top=int(torch.randint(0, img.height - img.width, (1,)).item()), left=0, height=img.width, width=img.width))
        self.random_square_width = transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=int(torch.randint(0, img.width - img.height, (1,)).item()), height=img.height, width=img.height))

        min_crop_scale = 0.5 if self.args.strong_aug_stage1 else 0.9
        if args.viton:
            width=192
            height=256
        elif args.viton_hd:
            width=768
            height=1024
        if args.MPV3D:
            width=320
            height=512
        self.width_base = width
        self.height_base = height

        target_size = max(width, height)
        
        padding_left = (target_size - width) // 2
        padding_top = (target_size - height) // 2
        padding_right = target_size - width - padding_left
        padding_bottom = target_size - height - padding_top
        # print('=======self.img_size',self.img_size)
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(
            #     self.img_size,
            #     scale=(min_crop_scale, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), padding_mode="edge"),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.cond_transform = transforms.Compose([
            # transforms.RandomResizedCrop(
            #     self.img_size,
            #     scale=(min_crop_scale, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), padding_mode="edge"),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.ref_transform = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            # transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), padding_mode="edge"),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomResizedCrop(
            #     (224, 224),
            #     scale=(min_crop_scale, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225]),
        ])
        self.ref_transform_mask = transforms.Compose([ # follow CLIP transform
            # transforms.RandomResizedCrop(
            #     (224, 224),
            #     scale=(min_crop_scale, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), padding_mode="edge"),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.tensor_transforms = transforms.Compose(
        [
            transforms.Normalize([0.5], [0.5]),
        ]
        )

    def add_mask_to_img(self, img, mask, img_key): #pil, pil
        if not img.size == mask.size:
            # import pdb; pdb.set_trace()
            # print(f'Reference image ({img_key}) size ({img.size}) is different from the mask size ({mask.size}), therefore try to resize the mask')
            mask = mask.resize(img.size) # resize the mask
        mask_array = np.array(mask)
        img_array = np.array(img)
        mask_array[mask_array < 127.5] = 0
        mask_array[mask_array > 127.5] = 1
        return Image.fromarray(img_array * mask_array), Image.fromarray(img_array * (1-mask_array)) # foreground, background

    def normalize_mask(self, mask):
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        return mask

    def augmentation(self, frame, transform1, transform2=None, state=None):
        if state is not None:
            torch.set_rng_state(state)
        frame_transform1 = transform1(frame) if transform1 is not None else frame
        if transform2 is None:
            return frame_transform1
        else:
            return transform2(frame_transform1)
    
    def get_metadata(self, idx):
        img_idx = self.get_image_index(idx)
        img_key = self.image_keys[img_idx]
        
        
        # (caption_sample, tag, start,
        #  end, _) = self.get_caption_and_timeinfo_wrapper(
        #     img_idx, cap_idx)
        # get image or video frames
        # frames: (T, C, H, W),  is_video: binary tag
        img = self.get_img(img_idx)
        # if isinstance(caption_sample, dict):
        #     caption = caption_sample["caption"]
        # else:
        #     caption = caption_sample
        #     caption_sample = None

        # preparing outputs
        meta_data = {}
        # meta_data['caption'] = caption  # raw text data, not tokenized
        meta_data['img_key'] = img_key
        meta_data['pose_img'] = None # setting pose to None
        # if self.args.combine_use_mask:
        #     meta_data['mask_img'] = self.get_cond(img_idx, 'masks')

        # ref and target image are the same 
        meta_data['ref_img_key'] = img_key
        if self.args.base:
            meta_data['reference_img'] = img 
        else:
            meta_data['reference_img'] = self.get_cloth(img_idx)
        if self.args.add_shape:
            meta_data['shape'] = self.get_shape(img_idx)
        if self.args.combine_use_mask:
            if self.args.base:
                meta_data['mask_img_ref'] = self.get_img_mask(img_idx)
            else:
                meta_data['mask_img_ref'] = self.get_cloth_mask(img_idx)
        meta_data['img'] = img
        if not self.args.no_smpl:
            meta_data['smpl'] = self.get_smpl(img_idx)
        meta_data['dp'] = self.get_dp(img_key)
        meta_data['uv'] = self.get_uvmap(img_key)
        meta_data['imap'] = self.get_imap(img_key)
        meta_data['correspondence'] = self.get_cor(img_key)
        # print('img',meta_data['img'])
        # print('dp',meta_data['dp'])
        return meta_data
    
    def get_img(self, img_idx):
        try:
            row = self.get_row_from_tsv(self.visual_tsv, img_idx)
            return self.str2img(row[-1])
        except Exception as e:
            raise ValueError(
                    f"{e}, in get_img()")
    def get_dp(self, img_key):
        # print('self.dp_file',self.dp_file)
        # /HOME/HOME/jisihui/VITON-hd-resized/try/densepose
        # dp_path=self.dp_file+'/'+img_key+'.npy'
        dp_name_list = os.listdir(self.dp_file)
        dp_paths = [ self.dp_file + '/' + dp_name for dp_name in dp_name_list]
        dp_paths.sort()
        dp_list=[]
        for dp_path in dp_paths:
            if not os.path.exists(dp_path):
                print('error!!')
            # print(dp_path)
            dp = F.interpolate(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0), (self.height_base, self.width_base), mode='bilinear').squeeze(0)
            dp = self.tensor_transforms(dp)
            # print('before==',dp.shape)
            #  torch.Size([2, 1024, 768])
            # print('before==',dp.unsqueeze(0).shape)
            # before== torch.Size([1, 2, 1024, 768])
            dp_list.append(dp.unsqueeze(0))
            
            # torch.Size([2, 1024, 768])
        dp_list = torch.cat(dp_list,dim=0)
        
        # print('dp_list0',dp_list.shape)
        # torch.Size([10, 2, 1024, 768])
        # torch.Size([20, 1024, 768])
        # torch.Size([2,10, 1024, 768])
        return dp_list
    
    def get_imap(self, img_key):
        # print('self.dp_file',self.dp_file)
        # /HOME/HOME/jisihui/VITON-hd-resized/try/densepose
        # dp_path=self.dp_file+'/'+img_key+'.npy'
        dp_name_list = os.listdir(self.imap_file)
        dp_paths = [ self.imap_file + '/' + dp_name for dp_name in dp_name_list]
        dp_paths.sort()
        dp_list=[]
        for dp_path in dp_paths:
            if not os.path.exists(dp_path):
                print('error!!')
            # print(dp_path)
            dp = F.interpolate(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0), (self.height_base, self.width_base), mode='bilinear')
            # print('before==',dp.shape)
            #  torch.Size([1, 1024, 768])
            # print('before==',dp.unsqueeze(0).shape)
            # before== torch.Size([1, 1, 1024, 768])
            dp_list.append(dp)
            
            # torch.Size([2, 1024, 768])
        dp_list = torch.cat(dp_list,dim=0)
        
        # print('dp_list',dp_list.shape)
        # torch.Size([20, 1024, 768])
        # torch.Size([2,10, 1024, 768])
        return dp_list
    
    def get_uvmap(self, img_key):
        # print('self.dp_file',self.dp_file)
        # /HOME/HOME/jisihui/VITON-hd-resized/try/densepose
        # dp_path=self.dp_file+'/'+img_key+'.npy'
        dp_name_list = os.listdir(self.dp_file)
        dp_paths = [ self.dp_file + '/' + dp_name for dp_name in dp_name_list]
        dp_paths.sort()
        dp_list=[]
        for dp_path in dp_paths:
            if not os.path.exists(dp_path):
                print('error!!')
            # print(dp_path)
            dp = F.interpolate(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0), (self.height_base, self.width_base), mode='bilinear')
            # print('before==',dp.shape)
            #  torch.Size([1, 1024, 768])
            # print('before==',dp.unsqueeze(0).shape)
            # before== torch.Size([1, 1, 1024, 768])
            dp_list.append(dp)
            
            # torch.Size([2, 1024, 768])
        dp_list = torch.cat(dp_list,dim=0)
        
        # print('dp_list',dp_list.shape)
        # torch.Size([20, 1024, 768])
        # torch.Size([2,10, 1024, 768])
        return dp_list
    
    def get_cor(self, img_key):
        # print('self.dp_file',self.dp_file)
        # /HOME/HOME/jisihui/VITON-hd-resized/try/densepose
        # dp_path=self.dp_file+'/'+img_key+'.npy'
        cor_name_list = os.listdir(self.cor_file)
        cor_paths = [ self.cor_file + '/' + cor_name for cor_name in cor_name_list]
        cor_paths.sort()
        cor_list=[]
        for cor_path in cor_paths:
            if not os.path.exists(cor_path):
                print('error!!')
            # print(dp_path)
            cor = torch.tensor(np.load(cor_path,allow_pickle=True))
            # [1024,768,2]
            # dp = F.interpolate(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0), (self.height_base, self.width_base), mode='bilinear').squeeze(0)
            # dp = self.tensor_transforms(dp)
            # print('before==',dp.shape)
            #  torch.Size([2, 1024, 768])
            # print('before==',dp.unsqueeze(0).shape)
            # before== torch.Size([1, 2, 1024, 768])
            cor_list.append(cor.unsqueeze(0))
            # [1,1024,768,2]
            
            # torch.Size([2, 1024, 768])
        cor_list = torch.cat(cor_list,dim=0).unsqueeze(1)
        # [10,1024,768,2]
        # torch.Size([10, 1, 1024, 768, 2])
        cor_list = cor_list.repeat(1, 10, 1, 1, 1)
        # print('cor_list',cor_list.shape)
        # [10, 10, 1024, 768, 2] 
        # cor_list = cor_list.repeat(2,1, 1, 1, 1, 1)

        # torch.Size([20, 1024, 768])
        # torch.Size([2,10, 1024, 768])
        return cor_list
    def get_shape(self, img_idx):
        try:
            # print(img_idx)
            row = self.get_row_from_tsv(self.shape_tsv, img_idx)
            return row[-1]
        except Exception as e:
            raise ValueError(
                    f"{e}, in get_shape()")
    def get_smpl(self, img_idx):
        try:
            row = self.get_row_from_tsv(self.smpl_tsv, img_idx)
            return self.str2img(row[-1])
        except Exception as e:
            raise ValueError(
                    f"{e}, in get_smpl()")
    def get_cloth(self, img_idx):
        # print(img_idx)
        try:
            row = self.get_row_from_tsv(self.cloth_tsv, img_idx)
            return self.str2img(row[-1])
        except Exception as e:
            raise ValueError(
                    f"{e}, in get_cloth()")
    def get_cloth_mask(self, img_idx):
        row = self.get_row_from_tsv(self.cloth_mask_tsv, img_idx)
        if len(row) == 3:
            image_key, buf, valid = row
            # assert image_key == self.image_keys[img_idx]
            if not valid:
                return None
            else:
                return self.str2img(buf)
        else:
            return self.str2img(row[1])
    def get_img_mask(self, img_idx):
        row = self.get_row_from_tsv(self.img_mask_tsv, img_idx)
        if len(row) == 3:
            image_key, buf, valid = row
            # assert image_key == self.image_keys[img_idx]
            if not valid:
                return None
            else:
                return self.str2img(buf)
        else:
            return self.str2img(row[1])
        

    def __len__(self):
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return min(self.args.max_train_samples, super().__len__())
            else:
                return super().__len__()
        else:
            if getattr(self.args, 'max_eval_samples', None):
                return min(self.args.max_eval_samples, super().__len__())
            else:
                return super().__len__()

    def __getitem__(self, idx):

        raw_data = self.get_metadata(idx)
        img = raw_data['img']
        if self.args.add_shape:
            shape = raw_data['shape']
        skeleton_img = raw_data['pose_img']
        reference_img = raw_data['reference_img']
        img_key = raw_data['img_key']
        ref_img_key = raw_data['ref_img_key']
        densepose = raw_data['dp']
        imap = raw_data['imap']
        uv = raw_data['uv']
        correspondence = raw_data['correspondence']
        # torch.Size([2, 1024, 768])
        # first check the size of the ref image
        ref_img_size = raw_data['reference_img'].size
        # if ref_img_size[0] > ref_img_size[1]: # width > height
        #     transform1 = self.random_square_width
        # elif ref_img_size[0] < ref_img_size[1]:
        #     transform1 = self.random_square_height
        # else:
        transform1 = None
        if not self.args.no_smpl:
            reference_img_controlnet = raw_data['smpl']
        else:
            reference_img_controlnet = raw_data['img']
        state = torch.get_rng_state()
        img = self.augmentation(img, transform1, self.transform, state)
        if skeleton_img is not None:
            skeleton_img = self.augmentation(skeleton_img, transform1, self.cond_transform, state)
        reference_img_controlnet = self.augmentation(reference_img_controlnet, transform1, self.transform, state) # controlnet path input
        if getattr(self.args, 'refer_clip_preprocess', None):
            reference_img = self.preprocesser(reference_img).pixel_values[0]  # use clip preprocess
        else:
            reference_img = self.augmentation(reference_img, transform1, self.ref_transform, state)

        reference_img_vae = reference_img_controlnet
        if self.args.combine_use_mask:
            mask_img_ref = raw_data['mask_img_ref']
            ### first resize mask to the img size
            mask_img_ref = mask_img_ref.resize(ref_img_size)

            assert not getattr(self.args, 'refer_clip_preprocess', None) # mask not support the CLIP process
            reference_img_mask = self.augmentation(mask_img_ref, transform1, self.ref_transform_mask, state)
            # reference_img_controlnet_mask = self.augmentation(mask_img_ref, transform1, self.cond_transform, state)  # controlnet path input
            if 'laion' in ref_img_key or 'deepfashion' in ref_img_key: # normailze mask for grounded sam mask
                reference_img_mask = self.normalize_mask(reference_img_mask)
                # reference_img_controlnet_mask = self.normalize_mask(reference_img_controlnet_mask)

            # apply the mask
            reference_img = reference_img * reference_img_mask# foreground

        # caption = raw_data['caption']
        outputs = {'img_key':img_key, 'label_imgs': img, 'uv':uv,'imap': imap, 'densepose':densepose,'correspondence':correspondence, 'reference_img': reference_img, 'reference_img_controlnet':reference_img_controlnet, 'reference_img_vae':reference_img_vae}
        if self.args.combine_use_mask:
            outputs['background_mask'] = (1 - reference_img_mask)
        if skeleton_img is not None:
            outputs.update({'cond_imgs': skeleton_img})
        if self.args.add_shape:
            outputs.update({'shape': shape})
        # print('==densepose',densepose.shape)
        # print('==imap',imap.shape)
        # torch.Size([20, 1024, 768])
        # torch.Size([10,1024, 768,2])
        # ==imap torch.Size([10, 1, 1024, 768])
        # ==densepose torch.Size([10, 2, 1024, 768])
        return outputs
