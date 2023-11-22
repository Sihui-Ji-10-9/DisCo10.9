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


class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        # print('-----~!!') yes
    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            meta=None,
            m=None,
            reso=None,
            inner_dim=None,
        ):
        # None
        # print('attn',attn)
        # attn CrossAttention(
        # _, c, h, w = hidden_states.shape
        # x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        
        batch_size, sequence_length, _ = hidden_states.shape
        # print('hidden_states.shape',hidden_states.shape)
        # torch.Size([20, 3072, 320])
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        print('encoder_hidden_states',encoder_hidden_states.shape)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # encoder_hidden_states torch.Size([20, 3072, 320])
        # print('k',key.shape)
        # k torch.Size([20, 3072, 320])
        # print('q',query.shape)
        # print('v',value.shape)
        # same
        # print('-----!!')
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

        query = attn.head_to_batch_dim(query)
        # torch.Size([160, 3072, 40])
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # attention_probs torch.Size([160, 3072, 3072])
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states torch.Size([20, 3072, 320])
        # from IPython import embed; embed()
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
