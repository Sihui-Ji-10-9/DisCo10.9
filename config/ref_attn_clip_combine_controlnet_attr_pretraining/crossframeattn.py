import os

import PIL.Image
import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2
from PIL import Image
from annotator.util import resize_image, HWC3
# from annotator.canny import CannyDetector
from annotator.openpose import OpenposeDetector
# from annotator.midas import MidasDetector
import decord
import math
from config.ref_attn_clip_combine_controlnet_attr_pretraining.utils import get_query_value,PosEmbedding
from Visualizer.visualizer import get_local
class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2,m=10,reso=(1024, 768)):
        self.unet_chunk_size = unet_chunk_size
        self.meta = {}
        self.m = m
        self.reso = reso
        # print('-----~!!') yes
    @get_local('attention_probs')
    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
        ):
        # None
        # print('attn',attn)
        # attn CrossAttention(
        
        # print('hidden_states.shape',hidden_states.shape)
        # torch.Size([20, 3072, 320])
        # 0.
        m = self.m
        reso = self.reso
        if self.meta !=None:
            x = hidden_states
            # print('=====m',m)
            # m=10
            # torch.Size([20, 3072, 320])
            # base:torch.Size([20, 320, 64, 48])
            _, multi,c  = x.shape
            h=int(2*math.sqrt(multi/3))
            w=int(0.5*math.sqrt(multi*3))
            x = rearrange(x, '(b m) (h w) c -> b m c h w', m=m,h=h,w=w)
            # torch.Size([2, 10, 320, 64, 48])
            b=x.shape[0]
            img_h, img_w = reso
            # img_h=1024
            # img_w=768
            # outs_q = []
            # outs_k = []
            # outs_m = []
            outs = []
            # pe = PosEmbedding(2, inner_dim//2)
            # poses = meta['poses']
            # K = meta['K']
            # print('densepose',meta['densepose'].shape)
            # print('imap',meta['imap'].shape)
            # densepose torch.Size([1, 10, 2, 1024, 768])
            # imap torch.Size([1, 10, 1, 1024, 768])
            # torch.Size([1, 10, 2, 1024, 768])
            # depths = meta['densepose'][:, :,:,1::2,1::2][:,:,0,:,:]
            # print('meta!!',self.meta.keys())
            depths = self.meta['uv'][:,:,0,:,:]
            depths = depths.repeat(2,1,1,1)
            imap = self.meta['imap'][:,:,0,:,:]
            imap = imap.repeat(2,1,1,1)
            # print('depths',depths.shape)
            # print('imap',imap.shape)
            # torch.Size([2, 10,1024, 768])
            # torch.Size([2, 10,1024, 768])

            # correspondence = meta['correspondence'][:, :,:,1::2,1::2]
            
            correspondence = self.meta['correspondence']
            correspondence = correspondence.repeat(2,1,1,1,1,1)
            # print('correspondence',correspondence.shape)
            # correspondence torch.Size([2, 10, 10, 1024, 768, 2])
            # correspondence torch.Size([1, 10, 10, 1024, 768, 2])
            # overlap_mask=meta['overlap_mask']

            for b_i in range(b):
                # x_outs_q=[]
                # x_outs_k=[]
                # x_outs_m=[]
                x_outs=[]
                for i in range(m):
                    # indexs = [j for j in range(m) if overlap_mask[b_i, i, j] and i!=j]
                    # if len(indexs)==0: # if the image does not have overlap with others, use the nearby images
                    #     if i==0:
                    #         indexs=[1]
                    #     elif i==m-1:
                    #         indexs=[m-2]
                    #     else:
                    #         indexs=[i-1, i+1]
                    indexs=[0]

                    xy_l = []
                    xy_r = []
                    x_right = []
                    # print('===',i)
                    # print('+++',b_i)
                    xy_l = correspondence[b_i:b_i+1, i, indexs]
                    # print('trans:xy_l',xy_l.shape)
                    # trans:xy_l torch.Size([1, 1, 1024,768, 2])
                    # [1,1,512,384,2]
                    xy_r = correspondence[b_i:b_i+1, indexs, i]
                    # [1,1,512,384,2]
                    x_left = x[b_i:b_i+1, i]
                    # print('trans:x_left',x_left.shape)
                    # trans:x_left torch.Size([1, 320, 64, 48])
                    x_right = x[b_i:b_i+1, indexs]  # bs, l, h, w, 
                    # torch.Size([1, 1, 320, 64, 48])
                    
                    # pose_l = poses[b_i:b_i+1, i]
                    # pose_r = poses[b_i:b_i+1, indexs]
                    
                    # pose_rel = torch.inverse(pose_l)[:, None]@pose_r
                    # depths guess [b m(1) 512 384]
                    # print('trans:depths',depths.shape)
                    # trans:depths torch.Size([2, 10, 1024, 768])
                    _depths=depths[b_i:b_i+1, indexs]
                    depth_query=depths[b_i:b_i+1, i]
                    _imap=imap[b_i:b_i+1, indexs]
                    imap_query=imap[b_i:b_i+1, i]
                    # print('trans:imap_query',imap_query.shape)
                    # print('trans:_imap',_imap.shape)
                    # trans:imap_query torch.Size([1, 1024, 768])
                    # trans:_imap torch.Size([1, 1, 1024, 768])
                    # print('trans:depth_query',depth_query.shape)
                    # trans:depth_query torch.Size([1, 1024, 768])
                    # _depths torch.Size([1,1, 512, 384])
                    # _K=K[b_i:b_i+1]
                    
                    query, key_value, key_value_xy, mask = get_query_value(
                        x_left, x_right, xy_l, xy_r, depth_query, _depths, imap_query, _imap, img_h, img_w, img_h, img_w)
                    # query [1, 320, 64, 48]
                    # key_value [1, 1,320, 64, 48]
                    # key_value_xy 1 1 64 48 1
                    # mask 1 1 64 48 
                    # print('----',key_value_xy*mask[..., None])
                    key_value_xy = rearrange(key_value_xy, 'b l h w c->(b h w) l c')

                    # 3072 1 1
                    # key_value_pe = pe(key_value_xy)
                    
                    # torch.Size([1, 320, 64, 48])
                    query = rearrange(query, 'b c h w->(b h w) c')[:, None]
                    # 3072 1 320 
                    key_value = rearrange(
                        key_value, 'b l c h w-> (b h w) l c')
                    # 3072 1 320
                    mask = rearrange(mask, 'b l h w -> (b h w) l')
                    # 3072 1
                    # 1111111111
                    # key_value = (key_value + key_value_pe)*mask[..., None]
                    mask = mask[..., None]
                    # 3072 1 320
                    key_value = key_value*mask
                    # query_pe = pe(torch.zeros(
                    #     query.shape[0], 1, 1, device=query.device))
                    # print('+++++++++',query_pe)
                    # 22222222222
                    # query = query + query_pe
                    query = query 
                    # 3072 1 320 
                    # out = self.transformer(query, key_value, query_pe)
                    # print('in,query',query.shape)
                    # in,query torch.Size([3072, 1, 320])
                    # query[:, 0]
                    #  3072 320 
                    out_m = rearrange(mask[:, 0], '(b h w) c -> b c h w', h=h, w=w)
                    out_q = rearrange(query[:, 0], '(b h w) c -> b c h w', h=h, w=w)
                    # 1 320 64 48 
                    out_k = rearrange(key_value[:, 0], '(b h w) c -> b c h w', h=h, w=w)
                    # 1 320 64 48
                    out_ = out_k*out_m+(~out_m)*out_q
                    # print('----',torch.sum(out_m!=0)/out_m.numel())
                    x_outs.append(out_)
                    # x_outs_q.append(out_q)
                    # x_outs_k.append(out_k)
                    # x_outs_m.append(out_m)
                x_outs=torch.cat(x_outs)
                # x_outs_q=torch.cat(x_outs_q)
                # x_outs_k=torch.cat(x_outs_k)
                # x_outs_m=torch.cat(x_outs_m)
                # x_outs torch.Size([10, 320, 64, 48])
                # 10 320 64 48
                outs.append(x_outs)
                # outs_q.append(x_outs_q)
                # outs_k.append(x_outs_k)
                # outs_m.append(x_outs_m)
                # print('x_outs_q',x_outs_q.shape)
                # x_outs torch.Size([10, 320, 64, 48])
            # out_q = torch.stack(outs_q)
            # out_k = torch.stack(outs_k)
            # out_m = torch.stack(outs_m)
            out = torch.stack(outs)
            # print('out',out.shape)
            # out torch.Size([2, 10, 320, 64, 48])
            out=rearrange(out, 'b m c h w -> (b m) (h w) c')
            # out_q=rearrange(out_q, 'b m c h w -> (b m) c h w')
            # out_k=rearrange(out_k, 'b m c h w -> (b m) c h w')
            # out_m=rearrange(out_m, 'b m c h w -> (b m) c h w')
            # torch.Size([20, 3072, 320])
            out = out.to(dtype=hidden_states.dtype)
            # out_q = out_q.to(dtype=hidden_states.dtype)
            # out_k = out_k.to(dtype=hidden_states.dtype)
            # out_m = out_m.to(dtype=hidden_states.dtype)

            # base:# torch.Size([20, 320, 64, 48])

        # -----
        batch_size, sequence_length, _ = hidden_states.shape
        # print('hidden_states.shape',hidden_states.shape)
        # torch.Size([20, 3072, 320])
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            # encoder_hidden_states = hidden_states
            if self.meta!=None:
                encoder_hidden_states = out
            else:
                encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        # print('encoder_hidden_states',encoder_hidden_states.shape)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # encoder_hidden_states torch.Size([20, 3072, 320])
        # print('k',key.shape)
        # k torch.Size([20, 3072, 320])
        # print('q',query.shape)
        # print('v',value.shape)
        # same
        # print('-----!!')
        '''
        # Sparse Attention
        if not is_cross_attention:
            # print('--',key.size()[0],self.unet_chunk_size)
            # -- 20 2
            video_length = key.size()[0] // self.unet_chunk_size
            # print('video_length',video_length)
            # video_length 10
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")
        '''
        query = attn.head_to_batch_dim(query)
        # torch.Size([160, 3072, 40])
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # if is_cross_attention:
        # attention_mask=None
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # from IPython import embed; embed()
        # *8
        # key.shape torch.Size([160, 3072, 40])
        # attention_probs torch.Size([160, 3072, 3072])
        hidden_states = torch.bmm(attention_probs, value)
        # 改！
        # hidden_states.shape torch.Size([160, 3072, 40])
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states torch.Size([20, 3072, 320])
        # from IPython import embed; embed()
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
