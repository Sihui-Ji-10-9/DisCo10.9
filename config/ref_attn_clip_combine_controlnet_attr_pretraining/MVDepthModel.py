import torch
import torch.nn as nn
from .modules import CPBlock, ImageEncodingBlock
from .utils import get_correspondence
from einops import rearrange

from diffusers.utils import BaseOutput, logging
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor

class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, config):
        super().__init__()
        self.overlap_filter=config['overlap_filter']
        self.unet = unet

        self.trainable_parameters = []
       
        self.cp_blocks_encoder = nn.ModuleList()
        for i in range(len(self.unet.down_blocks)):
            self.cp_blocks_encoder.append(CPBlock(
                self.unet.down_blocks[i].resnets[-1].out_channels, flag360=False))

        self.cp_blocks_mid = CPBlock(
            self.unet.mid_block.resnets[-1].out_channels, flag360=False)

        self.cp_blocks_decoder = nn.ModuleList()
        for i in range(len(self.unet.up_blocks)):
            self.cp_blocks_decoder.append(CPBlock(
                self.unet.up_blocks[i].resnets[-1].out_channels, flag360=False))
        self.condition_conv_in = nn.Conv2d(
            in_channels=5, out_channels=self.unet.conv_in.out_channels, kernel_size=3, stride=1, padding=1)
        self.condition_downblocks = nn.ModuleList([])
        stride=1
        for i in range(len(self.unet.down_blocks)):
            block = ImageEncodingBlock(
                in_channels=self.unet.conv_in.out_channels,
                out_channels=self.unet.down_blocks[i].resnets[0].in_channels,  stride=stride
            )
            self.condition_downblocks.append(block)
            if self.unet.down_blocks[i].downsamplers is not None:
                stride*=2
       
        self.condition_upblocks = nn.ModuleList([])
        channels=[1280, 1280, 1280, 640]
        for i in range(len(self.unet.up_blocks)):
            block = ImageEncodingBlock(
                in_channels=self.unet.conv_in.out_channels,
                out_channels=channels[i],  stride=stride
            )
            self.condition_upblocks.append(block)
            if self.unet.up_blocks[i].upsamplers is not None:
                stride//=2

        self.trainable_parameters +=[
            (list(self.cp_blocks_encoder.parameters())+
            list(self.cp_blocks_mid.parameters())+
            list(self.cp_blocks_decoder.parameters()), 1.0)
        ]

        self.trainable_parameters += [(list(self.condition_conv_in.parameters()) +
                                    list(self.condition_downblocks.parameters())+
                                    list(self.condition_upblocks.parameters()), 1.0)]

    def get_correspondence(self, cp_package):
        
        poses = cp_package['poses']
        K = cp_package['K']
        depths = cp_package['depths']
        b, m, h, w = depths.shape

        correspondence = torch.zeros(b, m, m, h, w, 2, device=depths.device)
        K = K[:, None].repeat(1, m, 1, 1)
        K = rearrange(K, 'b m h w -> (b m) h w')
        overlap_ratios=torch.zeros(b, m, m, device=depths.device)
        
        for i in range(m):
            pose_i = poses[:, i:i+1].repeat(1, m, 1, 1)
            depth_i = depths[:, i:i+1].repeat(1, m, 1, 1)
            pose_j = poses
            depth_i = rearrange(depth_i, 'b m h w -> (b m) h w')
            pose_j = rearrange(pose_j, 'b m h w -> (b m) h w')
            pose_i = rearrange(pose_i, 'b m h w -> (b m) h w')
            pose_rel = torch.inverse(pose_j)@pose_i
            point_ij, _ = get_correspondence(
                depth_i, pose_rel, K, None)  # bs, 2, hw
            point_ij = rearrange(point_ij, '(b m) h w c -> b m h w c', b=b)
            correspondence[:, i] = point_ij
            mask=(point_ij[:,:,:,:,0]>=0)&(point_ij[:,:,:,:,0]<w)&(point_ij[:,:,:,:,1]>=0)&(point_ij[:,:,:,:,1]<h)
            mask=rearrange(mask, 'b m h w -> b m (h w)')
            overlap_ratios[:,i]=mask.float().mean(dim=-1)
        for b_i in range(b):
            for i in range(m):
                for j in range(i+1,m):
                    overlap_ratios[b_i, i, j] = overlap_ratios[b_i, j, i]=min(overlap_ratios[b_i, i, j], overlap_ratios[b_i, j, i])
        overlap_mask=overlap_ratios>self.overlap_filter # filter image pairs that have too small overlaps
        cp_package['correspondence'] = correspondence
        cp_package['overlap_mask']=overlap_mask



    def forward(
        self,
        meta,
        latents_lr: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        prompt_embd: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in latents_lr.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(latents_lr.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.unet.config.center_input_sample:
            latents_lr = 2 * latents_lr - 1.0
        
        # 1. process timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latents_lr.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latents_lr.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latents_lr.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latents_lr.shape[1])

        t_emb = self.unet.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.unet.dtype)

        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)

        if self.unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.unet.config.class_embed_type == "timestep":
                class_labels = self.unet.time_proj(class_labels)

            class_emb = self.unet.class_embedding(class_labels).to(dtype=self.unet.dtype)
            emb = emb + class_emb
    
        # 2. pre-process
        if 'condition' in meta:
            condition=rearrange(meta['condition'], 'b m c h w -> (b m) c h w')
            condition_states=self.condition_conv_in(condition)
            condition_flag=True
        else:
            condition_flag=False

        # compute correspondence
        self.get_correspondence(meta)

        hidden_states=latents_lr
        # [1,20, 4, 64, 48]
        b, m, c, h_lr, w_lr = hidden_states.shape
        reso_lr = h_lr*8, w_lr*8
        cp_mask=torch.ones(m, m, device=hidden_states.device)
        for i in range(m):
            cp_mask[i, i]=0

        hidden_states = rearrange(hidden_states, 'b m c h w -> (b m) c h w')
        # torch.Size([1, 20, 973, 768])
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')
        
        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64

        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)
        
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if condition_flag: # Image condition 
                condition_states_i=self.condition_downblocks[i](condition_states)
                hidden_states+=condition_states_i


            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:

                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states, emb)
                   
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)

            if m>1:
                hidden_states = self.cp_blocks_encoder[i](
                    hidden_states, reso_lr, meta, m)
             

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid
        
        if condition_flag:
            condition_states_i=self.condition_downblocks[i](condition_states)
            hidden_states+=condition_states_i
        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)
        if m>1: # correspondence aware attention
            hidden_states = self.cp_blocks_mid(
                hidden_states, reso_lr, meta, m)

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]
            if condition_flag:
                condition_states_i=self.condition_upblocks[i](condition_states)
                hidden_states+=condition_states_i
            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
            if m>1:
                hidden_states = self.cp_blocks_decoder[i](
                    hidden_states, reso_lr, meta, m)
            
            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)

       
        return sample
