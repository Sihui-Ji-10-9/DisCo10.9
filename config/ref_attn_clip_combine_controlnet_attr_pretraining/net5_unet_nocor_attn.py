import torch
from utils.dist import synchronize, get_rank
from .crossframeattn_base import CrossFrameAttnProcessor
from config import *
from typing import Callable, List, Optional, Union

import inspect
from typing import Callable, List, Optional, Union

import torch
import random
from packaging import version
from transformers import CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModel
from transformers import AutoImageProcessor, AutoModel
from tqdm.auto import tqdm

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, DDPMScheduler, UniPCMultistepScheduler

from diffusers.utils import deprecate, PIL_INTERPOLATION
from diffusers.utils.import_utils import is_xformers_available

from visualizer import get_local
get_local.activate() # 激活装饰器

# from diffusers.models import UNet2DConditionModel
from .unet_2d_condition import UNet2DConditionModel
import PIL.Image
from .controlnet import ControlNetModel, MultiControlNetModel_MultiHiddenStates
# from PIL import Image
from utils.common import ensure_directory
from utils.dist import synchronize
from dinov2.dinov_2 import get_dinov2_model

from einops import rearrange
import imageio
from consistencydecoder import ConsistencyDecoder
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

class Net(nn.Module):
    def __init__(
        self, args
    ):
        super().__init__()
        self.args = args
        
        tr_noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_path, subfolder="scheduler")
        if self.args.eval_scheduler == "ddpm":
            noise_scheduler = DDPMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder="scheduler")
        elif self.args.eval_scheduler == "ddim":
            noise_scheduler = DDIMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder="scheduler")
        else:
            noise_scheduler = PNDMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder="scheduler")
        # tokenizer = CLIPTokenizer.from_pretrained(
        #     args.pretrained_model_path, subfolder="tokenizer")
        # print('====',args.pretrained_model_path)
        
        print('Loading CLIP image encoder')
        # feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_path, subfolder="feature_extractor")
        feature_extractor = AutoImageProcessor.from_pretrained('/home/nfs/jsh/DisCo/huggingface/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c', crop_size={'height': args.img_full_size[0], 'width': args.img_full_size[0]})
        print(f"Loading pre-trained image_encoder from {args.pretrained_model_path}/image_encoder")
        # clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="image_encoder")
        
        print(f'Loading DINOv2 image encoder, version {self.args.dinov2_version}')
        # clip_image_encoder = AutoModel.from_pretrained('facebook/dinov2-large')
        clip_image_encoder = AutoModel.from_pretrained('/home/nfs/jsh/DisCo/huggingface/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c')
        # clip_image_encoder = AutoModel.from_pretrained('/mnt_group/yuer.qian/pretrain_model/huggingface/dinov2_tryon_19m_20ep_vitl14')

        # dinov2_image_encoder = get_dinov2_model(self.args.dinov2_model_path, version=self.args.dinov2_version, pretrained=False)
        
        # clip_image_encoder = torch.hub.load('facebookresearch/dinov2', self.args.dinov2_version, pretrained=False)
        # clip_model_path = os.path.join(self.args.dinov2_model_path, self.args.dinov2_version + '_pretrain.pth')
        # clip_image_encoder.load_state_dict(torch.load(clip_model_path), strict=True)


        # self.dinov2_head = nn.Linear(1536, 768)
        # self.dinov2_head.requires_grad_(True)

        print(f"Loading pre-trained vae from {args.pretrained_model_path}/vae")
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_path, subfolder="vae")
        if self.args.use_con_dec:
            decoder_consistency = ConsistencyDecoder(device="cuda:0") # Model size: 2.49 GB
        print(f"Loading pre-trained unet from {self.args.pretrained_model_path}/unet")
        unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_path, subfolder="unet")

        if hasattr(noise_scheduler.config, "steps_offset") and noise_scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {noise_scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {noise_scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(self.noise_scheduler.config)
            new_config["steps_offset"] = 1
            noise_scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(noise_scheduler.config, "clip_sample") and noise_scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {noise_scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(noise_scheduler.config)
            new_config["clip_sample"] = False
            noise_scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(self.unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        
        # Modify input layer to have 1 additional input channels (pose)
        unet.base_conv_in = unet.conv_in
        weights = unet.conv_in.weight.clone()
        with torch.no_grad():
            unet.base_conv_in.weight[:,:] = weights # original weights
        # print('weights',weights.shape)
        # torch.Size([320, 4, 3, 3])
        unet.conv_in = nn.Conv2d(6, weights.shape[0], kernel_size=3, padding=(1, 1)) # input noise + n poses
        with torch.no_grad():
            # print('unet.conv_in.weight',unet.conv_in.weight.shape)
            # unet.conv_in.weight torch.Size([320, 7, 3, 3])
            # print('[:, :4]',unet.conv_in.weight[:, :4].shape)
            # torch.Size([320, 4, 3, 3])
            unet.conv_in.weight[:, :4] = weights # original weights
            # print('[:, 3:]',unet.conv_in.weight[:, 3:].shape)
            # torch.Size([320, 4, 3, 3])
            unet.conv_in.weight[:, 4:] = torch.zeros(unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
        self.text2video_attn_proc = CrossFrameAttnProcessor(unet_chunk_size=2)
        
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                print("xformers is not available, therefore not enabled")
        if self.args.use_cf_attn:
            # print('yes') yes
            unet.set_attn_processor(processor=self.text2video_attn_proc)
        
        tokenizer = CLIPTokenizer.from_pretrained(self.args.sd15_path+ "/tokenizer")
        self.tokenizer = tokenizer
        print(f"Loading pre-trained text_encoder from {self.args.sd15_path}/text_encoder")
        text_encoder = CLIPTextModel.from_pretrained(self.args.sd15_path + "/text_encoder")
        self.text_encoder = text_encoder
    
        '''
        if args.ref_null_caption:
            tokenizer = CLIPTokenizer.from_pretrained(self.args.sd15_path, subfolder="tokenizer")
            self.tokenizer = tokenizer
            print(f"Loading pre-trained text_encoder from {self.args.sd15_path}/text_encoder")
            text_encoder = CLIPTextModel.from_pretrained(self.args.sd15_path + "/text_encoder")
            self.text_encoder = text_encoder
            unet_sd15 = UNet2DConditionModel.from_pretrained(self.args.sd15_path, subfolder="unet")
            controlnet_background = ControlNetModel.from_unet(unet=unet_sd15, args=self.args, use_sd_vae=True) # initialize ref controlnet path from the SD pretrained model
            del unet_sd15
        else: # initialize controlnet from the variation SD
            controlnet_background = ControlNetModel.from_unet(unet=unet, args=self.args, use_sd_vae=True)

        if self.args.gradient_checkpointing:
            # controlnet_pose.enable_gradient_checkpointing()
            controlnet_background.enable_gradient_checkpointing()
        # controlnet_unit = MultiControlNetModel_MultiHiddenStates([controlnet_pose, controlnet_background])
        controlnet_unit = controlnet_background
        '''
        self.tr_noise_scheduler = tr_noise_scheduler
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        if self.args.use_con_dec:
            self.decoder_consistency = decoder_consistency
        # self.controlnet = controlnet_unit
        self.unet = unet
        self.feature_extractor = feature_extractor
        self.clip_image_encoder = clip_image_encoder
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.drop_text_prob = args.drop_text
        self.device = torch.device('cpu')
        self.dtype = torch.float32

        self.scale_factor = self.args.scale_factor
        self.guidance_scale = self.args.guidance_scale
        self.controlnet_conditioning_scale = getattr(self.args, "controlnet_conditioning_scale", 1.0)
        self.controlnet_conditioning_scale_cond = getattr(self.args, "controlnet_conditioning_scale_cond", 1.0)
        self.controlnet_conditioning_scale_ref = getattr(self.args, "controlnet_conditioning_scale_ref", 1.0)

        
        if getattr(self.args, 'combine_clip_local', None) and not getattr(self.args, 'refer_clip_proj', None): # not use clip pretrained visual projection (but initialize from it)
            # self.refer_clip_proj = torch.nn.Linear(clip_image_encoder.visual_projection.in_features, clip_image_encoder.visual_projection.out_features, bias=False)
            # self.refer_clip_proj.load_state_dict(clip_image_encoder.visual_projection.state_dict())
            self.refer_clip_proj = nn.Sequential(
                                nn.Linear(1024, 768),
                                nn.LayerNorm(768))
            self.refer_clip_proj.requires_grad_(True)
        if args.add_shape:
            self.cc_projection1 = nn.Linear(10,1000)
            self.relu = nn.ReLU()
            self.cc_projection2 = nn.Linear(1000,768)
            nn.init.eye_(list(self.cc_projection1.parameters())[0][:1000,:1000])
            nn.init.zeros_(list(self.cc_projection1.parameters())[1])
            self.cc_projection1.requires_grad_(True)
            nn.init.eye_(list(self.cc_projection2.parameters())[0][:768,:768])
            nn.init.zeros_(list(self.cc_projection2.parameters())[1])
            self.cc_projection2.requires_grad_(True)
        # self.conv_layer = nn.Conv2d(7, 4, kernel_size=3, stride=1, padding=1)


    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def init_ddpm(self):
        self.freeze_pretrained_part_in_ddpm()
        return

    def freeze_pretrained_part_in_ddpm(self):
        if self.args.freeze_unet:
            #b self.unet.eval()
            param_unfreeze_num = 0
            if self.args.unet_unfreeze_type == 'crossattn-kv':
                for param_name, param in self.unet.named_parameters(): # only to set attn2 k, v to be requires_grad
                    if 'transformer_blocks' not in param_name:
                        param.requires_grad_(False)
                    elif not ('attn2.to_k' in param_name or 'attn2.to_v' in param_name):
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == 'crossattn':
                for param_name, param in self.unet.named_parameters(): # only to set attn2 k, v to be requires_grad
                    if 'transformer_blocks' not in param_name:
                        param.requires_grad_(False)
                    elif not 'attn2' in param_name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == 'transblocks':
                for param_name, param in self.unet.named_parameters():
                    if 'transformer_blocks' not in param_name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1
                    if 'conv_in' in param_name:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == 'all':
                for param_name, param in self.unet.named_parameters():
                    param.requires_grad_(True)
                    param_unfreeze_num += 1

            else: # freeze all the unet
                print('Unmatch to any option, freeze all the unet')
                self.unet.eval()
                self.unet.requires_grad_(False)

            print(f"Mode [{self.args.unet_unfreeze_type}]: There are {param_unfreeze_num} modules in unet to be set as requires_grad=True.")

        self.vae.eval()
        self.clip_image_encoder.eval()
        self.vae.requires_grad_(False)
        self.clip_image_encoder.requires_grad_(False)
        if hasattr(self, 'text_encoder'):
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.unet.parameters()).dtype
        self.clip_image_encoder.float()
        if hasattr(self, 'text_encoder'):
            self.text_encoder.float()
        # self.refer_clip_proj.float()
        # self.vae.float()
        return model_converted
    
    def half(self, *args, **kwargs):
        super().half(*args, **kwargs)
        self.dtype = torch.float16
        self.clip_image_encoder.float()
        if hasattr(self, 'text_encoder'):
            self.text_encoder.float()
        # self.refer_clip_proj.float()
        # self.vae.float()
        return

    def train(self, *args):
        super().train(*args)
        self.freeze_pretrained_part_in_ddpm()

    def image_encoder(self, image):
        b, c, h, w = image.size()
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.scale_factor
        latents = latents.to(dtype=self.dtype)
        return latents

    def image_decoder(self, latents):
        latents = 1/self.scale_factor * latents
        if self.args.use_con_dec:
            dec = self.decoder_consistency(latents)
        else:
            dec = self.vae.decode(latents).sample
        image = (dec / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, inputs):
        outputs = dict()
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.args.seed)
        inputs['generator'] = generator

        if self.training:
            assert self.args.stepwise_sample_depth <= 0
            # outputs = self.forward_train(inputs, outputs)
            outputs = self.forward_train_multicontrol(inputs, outputs)
        elif 'enc_dec_only' in inputs and inputs['enc_dec_only']:
            outputs = self.forward_enc_dec(inputs, outputs)
        else:
            assert self.args.stepwise_sample_depth <= 0
            # outputs = self.forward_sample(inputs, outputs)
            outputs = self.forward_sample_multicontrol(inputs, outputs)
        removed_key = inputs.pop("generator", None)
        return outputs

    def text_encode(
            self, prompt, num_images_per_prompt=1,
            do_classifier_free_guidance=False, negative_prompt=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(self.device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat(
                [uncond_embeddings, text_embeddings]
                )
        text_embeddings = text_embeddings.to(dtype=self.dtype)
        return text_embeddings


    def clip_encode_image_global(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False): # clip global feature
        dtype = next(self.clip_image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=dtype)
        image_embeddings = self.clip_image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.dtype)


    def clip_encode_image_local(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False): # clip local feature
        dtype = next(self.clip_image_encoder.parameters()).dtype
        # print('==0dtype',dtype)
        # torch.float32
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            # print('=========================!') no use
        image = image.to(device=self.device, dtype=dtype)
        last_hidden_states = self.clip_image_encoder(image).last_hidden_state
        last_hidden_states_norm = last_hidden_states #self.clip_image_encoder.vision_model.post_layernorm(last_hidden_states)
        # print('====',last_hidden_states_norm.shape) === torch.Size([4, 257, 1024]) 
        if self.args.refer_clip_proj: # directly use clip pretrained projection layer
            image_embeddings = self.clip_image_encoder.visual_projection(last_hidden_states_norm)
        else:
            image_embeddings = self.refer_clip_proj(last_hidden_states_norm.to(dtype=self.dtype))
        # image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.dtype)    
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta=0.0):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
            self, batch_size, num_channels_latents,
            height, width, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        # print('===batch_size',batch_size)
        # ===batch_size 10
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, list):
                # print('===1')
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], dtype=self.dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(self.device)
            else:
                # print('===2')
                # here!
                latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=self.device, dtype=self.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def prepare_latents_fix(
            self, batch_size, num_channels_latents,
            height, width, generator, latents=None):
        shape = (1, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        # print('===batch_size',batch_size)
        # ===batch_size 10
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, list):
                # print('===1')
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], dtype=self.dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(self.device)
            else:
                # print('===2')
                # here!
                latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
                latents = latents.repeat(batch_size,1,1,1)
                # from IPython import embed; embed()
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=self.device, dtype=self.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents
    
    def forward_train_multicontrol(self, inputs, outputs):
        # # use CFG
        # if self.args.drop_ref > 0:
        #     p = random.random()
        #     if p <= self.args.drop_ref: # dropout ref image
        #         inputs['reference_img'] = torch.zeros_like(inputs['reference_img'])
        loss_target = self.args.loss_target
        image = inputs['label_imgs']  # (B, C, H, W)
        ref_image = inputs['reference_img']
        densepose = inputs['densepose']
        bsz = image.shape[0]

        if self.args.ref_null_caption:
            text = inputs['input_text']
            if random.random() < self.args.drop_text: # drop text w.r.t the prob
                text = ["" for i in text]
            z_text = self.text_encode(text)

        # text SD input --> reference image input (clip global embedding)
        if self.args.combine_clip_local:
            refer_latents = self.clip_encode_image_local(ref_image).to(dtype=self.dtype)
        else:
            refer_latents = self.clip_encode_image_global(ref_image).to(dtype=self.dtype)
        start_code, latents_list = self.invert(ref_image,
                                               refer_latents,
                                                guidance_scale=7.5,
                                                num_inference_steps=50,
                                                return_intermediates=True)
        if self.args.add_shape:
            shape =torch.tensor([eval(s) for s in inputs['shape']])
            shape =shape[:,None,:].to(memory_format=torch.contiguous_format).float()
            shape = shape.to(device=self.device,dtype=self.dtype)
            with torch.enable_grad():
                shape = self.cc_projection1(shape)
                # print('shape1',shape.shape)
                # shape1 torch.Size([64, 1, 1000])
                shape = self.relu(shape)
                shape = self.cc_projection2(shape)
                # print('shape2',shape.shape)
                # shape2 torch.Size([64, 1, 768])
                refer_latents = torch.cat([refer_latents,shape],dim=1)
                # print('shape3',refer_latents.shape)
                # shape3 torch.Size([64, 258, 768])
        latents = self.image_encoder(image)
        latents = latents.to(dtype=self.dtype)
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0, self.tr_noise_scheduler.num_train_timesteps,
            (bsz,), device=latents.device)
        timesteps = timesteps.long()
        if self.args.debug_seed:
            print(f"rank {get_rank()}: noise 0 mean {torch.sum(noise[0])}, noise 1 mean {torch.sum(noise[1])}")
            print(f"timestep 0 {timesteps[0]}, timestep 1 {timesteps[1]}")
        noisy_latents = self.tr_noise_scheduler.add_noise(latents, noise, timesteps)


        # TODO: @tan, change cond_imgs in dataloadser to pose or other conditions.
        # controlnet_image = inputs["cond_imgs"].to(dtype=self.dtype)
        if self.args.refer_sdvae:
            reference_latents_controlnet = self.image_encoder(inputs["reference_img_controlnet"]) # controlnet path input
            reference_latents_controlnet = reference_latents_controlnet.to(dtype=self.dtype)
        else:
            reference_latents_controlnet = inputs["reference_img_controlnet"]
        # controlnet_image = [inputs["cond_imgs"], reference_latents_controlnet]  # [pose image, ref image]
        controlnet_image = reference_latents_controlnet  # [ref image]
        # smpl = reference_latents_controlnet.clone().detach()
        # Concatenate pose with noise
        _, _, h, w = noisy_latents.shape
        # print('noisy_latents',noisy_latents.shape)
        # torch.Size([64, 4, 32, 32])
        # print('smpl',smpl.shape)
        # torch.Size([64, 3, 256, 256])
        # print('densepose',densepose.shape)
        # densepose torch.Size([64, 2, 1024, 768])
        pose_input = F.interpolate(densepose, (h,w)).cuda().to(self.dtype)
        # print('pose_input',pose_input.shape)
        # pose_input torch.Size([64, 2, 32, 32])
        # print('smpl2',smpl_input.shape)
        # smpl2 torch.Size([64, 3, 32, 32])
        noisy_latents = torch.cat((noisy_latents.cuda(),pose_input), 1)
        # noisy_latents = self.conv_layer(noisy_latents)
        # print('noisy_latents',noisy_latents.shape)
        # noisy_latents torch.Size([64, 6, 32, 32])
        '''
        # controlnet get the input of (a. ref image clip embedding; b. pose cond image)
        if self.args.ref_null_caption:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents, timesteps, z_text, # reference controlnet path use null caption
                controlnet_cond=controlnet_image, return_dict=False)
        else:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents, timesteps, refer_latents, # both controlnet path use the refer latents
                controlnet_cond=controlnet_image, return_dict=False)
        '''
        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=refer_latents # refer latents
        ).sample

        if loss_target == "x0":
            target = latents
            x0_pred = self.tr_noise_scheduler.remove_noise(noisy_latents, model_pred, timesteps)
            loss = F.mse_loss(x0_pred.float(), target.float(), reduction="mean")
        else:
            if self.tr_noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif self.tr_noise_scheduler.prediction_type == "v_prediction":
                target = self.tr_noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.tr_noise_scheduler.prediction_type}")
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        outputs['loss_total'] = loss
        return outputs


    def prepare_image(
        self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image
    
    @torch.no_grad()
    def image2latent(self, image):
        # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device=self.device)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents
    
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps, 999)
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.noise_scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0
    
    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        refer_latents,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        batch_size = image.shape[0]
        '''
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        '''

        # text embeddings
        # text_input = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=77,
        #     return_tensors="pt"
        # )
        '''
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        print("input text embeddings :", text_embeddings.shape)
        #  torch.Size([1, 77, 768])
        '''
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)
        # exit()
        '''
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
        text_embeddings = text_embeddings.to(dtype = self.dtype)
        '''
        print("latents shape: ", latents.shape)
        # torch.Size([1, 4, 64, 48])
        
        # interative sampling
        self.noise_scheduler.set_timesteps(self.args.num_inference_steps, device=self.device)
        print("Valid timesteps: ", reversed(self.noise_scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.noise_scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t,encoder_hidden_states=refer_latents,is_invert=True).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
    @torch.no_grad()
    def forward_sample_multicontrol(self, inputs, outputs):
        gt_image = inputs['label_imgs']
        # print('gt_image',gt_image.shape)
        # torch.Size([3, 3, 256, 256])
        b, c, h, w = gt_image.size()
        ref_image = inputs['reference_img']
        img_key = inputs['img_key']
        
        # for img_name in img_key:
        # print('!!!',ref_image.shape)
        # print('===',img_key)
        # ['00008_00.jpg', '00035_00.jpg', '00067_00.jpg']
        densepose = inputs['densepose']
        correspondence = inputs['correspondence']
        # print('!',densepose.shape) torch.Size([1, 20, 1024, 768])
        # print('!!',ref_image.shape) torch.Size([1, 3, 256, 192])

        # torch.Size([2, 1024, 768])

        # print('1---',ref_image.shape)
        # torch.Size([10, 3, 512, 384])
        # print('2---',len(denseposes))
        # 10
        # print('3---',denseposes[0].shape)
        # torch.Size([10, 2, 1024, 768])
        # 1--- torch.Size([3, 3, 224, 224])
        # 2--- torch.Size([3, 2, 1024, 768])
        # print('!!!',ref_image.shape) torch.Size([10, 3, 224, 224])
        # torch.Size([1, 3, 512, 384])      
        do_classifier_free_guidance = self.guidance_scale > 1.0
        # print('do_classifier_free_guidance',do_classifier_free_guidance) True
        '''
        if not self.args.use_dinov2:
            if self.args.combine_clip_local:
                refer_latents = self.clip_encode_image_local(ref_image).to(dtype=self.dtype)
                # torch.Size([5, 257, 768])
            else:
                refer_latents = self.clip_encode_image_global(ref_image).to(dtype=self.dtype)
        else:
            refer_latents = self.dinov2_encode_image(ref_image).to(dtype=self.dtype)
        # print(refer_latents1.shape) ([5, 257, 768])
        # print(refer_latents.shape) ([5, 257, 768])
        # from IPython import embed; embed()
        '''
        
        if self.args.combine_clip_local:
            refer_latents = self.clip_encode_image_local(ref_image, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)
        else:
            refer_latents = self.clip_encode_image_global(ref_image, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)
        # -------                
        # invert the source image
        source_prompt = ""
        target_prompt = ""
        prompts = [source_prompt, target_prompt]
        
        start_code, latents_list = self.invert(ref_image,
                                               refer_latents,
                                                guidance_scale=7.5,
                                                num_inference_steps=50,
                                                return_intermediates=True)
        # start_code = start_code.expand(len(prompts), -1, -1, -1)
        # start_code torch.Size([2, 4, 64, 64])

        
        # refer_latents = refer_latents.repeat(10,1,1)
        # print('refer_latents',refer_latents.shape)
        # torch.Size([2, 973, 768])
        # torch.Size([20, 973, 768])
        refer_latents = torch.cat([repeat(refer_latents[0, :, :], "c k -> f c k", f=10),
                                   repeat(refer_latents[1, :, :], "c k -> f c k", f=10)])
        # torch.Size([20, 235, 768])
        # torch.Size([20, 973, 768])
        if self.args.ref_null_caption: # test must use null caption
            text = inputs['input_text']
            text = ["" for i in text]
            text_embeddings = self.text_encode(
                text, num_images_per_prompt=self.args.num_inf_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=None)
        if self.args.add_shape:
            shape =torch.tensor([eval(s) for s in inputs['shape']])
            shape =shape[:,None,:].to(memory_format=torch.contiguous_format).float()
            shape = shape.to(device=self.device,dtype=self.dtype)
            if do_classifier_free_guidance:
                zero_shape = torch.zeros_like(shape)
                shape = torch.cat([zero_shape, shape])
                shape = shape.to(device=self.device,dtype=self.dtype)
            # print('=====',shape.shape,refer_latents.shape) 
            # torch.Size([20, 1, 10]) torch.Size([20, 257, 768])
            with torch.enable_grad():
                shape = self.cc_projection1(shape)
                # print('shape1',shape.shape)
                # shape1 torch.Size([20, 1, 1000])
                shape = self.relu(shape)
                shape = self.cc_projection2(shape)
                # print('shape2',shape.shape)
                # shape2 torch.Size([20, 1, 768])
                refer_latents = torch.cat([refer_latents,shape],dim=1)
                # print('shape3',refer_latents.shape)
                # shape3 torch.Size([20, 258, 768])


        # Prepare conditioning image
        controlnet_conditioning_scale = self.controlnet_conditioning_scale
        controlnet_conditioning_scale_cond = self.controlnet_conditioning_scale_cond
        controlnet_conditioning_scale_ref = self.controlnet_conditioning_scale_ref
        # image_pose = self.prepare_image(
        #     image=inputs['cond_imgs'].to(dtype=self.dtype),
        #     width=w,
        #     height=h,
        #     batch_size=b * self.args.num_inf_images_per_prompt,
        #     num_images_per_prompt=self.args.num_inf_images_per_prompt,
        #     device=self.device,
        #     dtype=self.controlnet.dtype,
        #     do_classifier_free_guidance=do_classifier_free_guidance,
        # )

        # Prepare ref image
        if self.args.refer_sdvae:
            reference_latents_controlnet = self.image_encoder(inputs["reference_img_controlnet"])
            reference_latents_controlnet = reference_latents_controlnet.to(dtype=self.dtype)
        else:
            reference_latents_controlnet = inputs['reference_img_controlnet'].to(dtype=self.dtype)
        # print('reference_latents_controlnet',reference_latents_controlnet.shape)
        # pose = preprocess(pose) #处理维度
        reference_latents_controlnet = self.prepare_image(
            image=reference_latents_controlnet,
            width=w,
            height=h,
            batch_size=b * self.args.num_inf_images_per_prompt,
            num_images_per_prompt=self.args.num_inf_images_per_prompt,
            device=self.device,
            dtype=torch.float16,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        # smpl= reference_latents_controlnet.clone().detach()

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(
            self.args.num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        gen_height = h
        gen_width = w
        generator = inputs['generator']

        latents0 = self.prepare_latents(
            b * self.args.num_inf_images_per_prompt,
            num_channels_latents,
            gen_height,
            gen_width,
            generator,
            latents=None,
        )
        latents = self.prepare_latents_fix(
            b * self.args.num_inf_images_per_prompt,
            num_channels_latents,
            gen_height,
            gen_width,
            generator,
            latents=None,
        )
        
        # print('latents',latents.shape)
        # torch.Size([1, 4, 32, 24])
        # torch.Size([1, 4, 64, 48])
        latents = start_code.repeat(1,10,1,1,1)
        # latents = latents.repeat(1,10,1,1,1)

        # print('latents',latents.shape)
        # torch.Size([10, 4, 32, 24])
        # torch.Size([1,10, 4, 64, 48])
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)

        # Denoising loop
        num_warmup_steps = len(timesteps) - self.args.num_inference_steps * self.noise_scheduler.order
        with self.progress_bar(total=self.args.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                print('latent_model_input',latent_model_input.shape) 
                # torch.Size([20, 4, 32, 24])
                # torch.Size([20, 4, 64, 48])
                # torch.Size([2, 10, 4, 64, 48]) 
                # Add pose to noisy latents
                _, _, _, h, w = latent_model_input.shape
                # densepose torch.Size([10, 2, 1024, 768])
                # print('densepose',densepose.shape)
                # torch.Size([1, 10, 2, 1024, 768])
                if do_classifier_free_guidance:
                    # print('---',torch.zeros(densepose.shape).shape)
                    # --- torch.Size([1, 10, 2, 1024, 768])
                    pose_input = torch.cat([torch.zeros(densepose.shape).cuda(), densepose])
                    # print('pose_input',pose_input.shape)
                    # torch.Size([20, 2, 1024, 768]) 
                    # torch.Size([2, 10, 2, 1024, 768])
                    # from IPython import embed; embed()
                else:
                    pose_input = torch.cat([densepose, densepose])
                # from IPython import embed; embed()
                bb = pose_input.shape[0]
                pose_input = rearrange(pose_input, 'b m c h w -> (b m) c h w')
                pose_input= F.interpolate(pose_input, (h,w)).cuda().to(dtype=self.dtype)
                pose_input = rearrange(pose_input, '(b m) c h w -> b m c h w', b=bb)
                # print('pose_input',pose_input.shape) 
                # torch.Size([20, 2, 32, 24])
                # torch.Size([2, 10, 2, 64, 48])
                latent_model_input = torch.cat((latent_model_input.cuda(), pose_input), 2)
                # print('latent_model_input',latent_model_input.shape)
                # torch.Size([20, 6, 32, 24])
                # torch.Size([20, 6, 64, 48])  
                # torch.Size([2, 10, 6, 64, 48])
                '''
                # controlnet(s) inference
                if self.args.ref_null_caption: # null caption input for ref controlnet path
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=reference_latents_controlnet,
                        conditioning_scale=controlnet_conditioning_scale_ref,
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=refer_latents,
                        controlnet_cond=reference_latents_controlnet,
                        conditioning_scale=controlnet_conditioning_scale_ref,
                        return_dict=False,
                    )
                '''

                # predict the noise residual
                # latent_model_input torch.Size([2, 10, 6, 64, 48])
                # refer_latents  torch.Size([20, 973, 768])
                

                # inference the synthesized image with MasaCtrl
                STEP = 4
                LAYPER = 10

                # hijack the attention module
                editor = MutualSelfAttentionControl(STEP, LAYPER)
                regiter_attention_editor_diffusers(self, editor)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=refer_latents,
                    meta=inputs).sample.to(dtype=self.dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # print('==after',latents.shape)
                # torch.Size([10, 4, 32, 24])
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                    progress_bar.update()

        # Post-processing
        # print('===',latents.shape)
        # torch.Size([10, 4, 32, 24])
        # === torch.Size([1, 10, 4, 64, 48])
        latents = rearrange(latents, 'b m c h w -> (b m) c h w')
        gen_img = self.image_decoder(latents)
        gen_img = gen_img.detach().cpu()
        print('gen_img',gen_img.shape)
        # torch.Size([10, 3, 256, 192])
        outputs['logits_imgs'] = gen_img
        cache = get_local.cache # ->  {'your_attention_function': [attention_map]}
        return outputs


    @torch.no_grad()
    def forward_enc_dec(self, inputs, outputs):
        image = inputs['label_imgs']
        latent = self.image_encoder(image)
        gen_img = self.image_decoder(latent)
        outputs['logits_imgs'] = gen_img
        return outputs

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs


def inner_collect_fn(args, inputs, outputs, log_dir, global_step, eval_save_filename='eval_visu'):
    
    rank = get_rank()
    if rank == -1:
        splice = ''
    else:
        splice = '_' + str(rank)
    if global_step <= 0:
        eval_log_dir = os.path.join(log_dir, eval_save_filename)
    else:
        eval_log_dir = os.path.join(log_dir, 'eval_step_%d' % (global_step))
    ensure_directory(eval_log_dir)

    gt_save_path = os.path.join(eval_log_dir, 'gt')
    ensure_directory(gt_save_path)
    pred_save_path = os.path.join(eval_log_dir, f'pred_gs{args.guidance_scale}_scale-cond{args.controlnet_conditioning_scale_cond}-ref{args.controlnet_conditioning_scale_ref}')
    ensure_directory(pred_save_path)
    cond_save_path = os.path.join(eval_log_dir, 'cond')
    ensure_directory(cond_save_path)
    ref_save_path = os.path.join(eval_log_dir, 'ref')
    ensure_directory(ref_save_path)
    ref_control_save_path = os.path.join(eval_log_dir, 'ref_control')
    ensure_directory(ref_control_save_path)

    synchronize()
    if rank in [-1, 0]:
        logger.warning(eval_log_dir)

        # Save Model Setting
        type_output = [int, float, str, bool, tuple, dict, type(None), ]
        setting_output = {item: getattr(args, item) for item in dir(args) if
                          type(getattr(args, item)) in type_output and not item.startswith('__')}
        data2file(setting_output, os.path.join(eval_log_dir, 'Model_Setting.json'))

    dl = {**inputs, **{k: v for k, v in outputs.items() if k.split('_')[0] == 'logits'}}
    # print(dl['logits_imgs'].shape)
    # torch.Size([10, 3, 256, 192])
    # print(dl.keys())
    # dict_keys(['img_key', 'label_imgs', 'densepose', 'reference_img', 'reference_img_controlnet', 'reference_img_vae', 'background_mask', 'logits_imgs'])
    ## WT DEBUG
    # print('just for debug')
    # if 'cond_img_pose' in dl:
    #     del dl['cond_img_pose']
    #     del dl['cond_img_attr']
    ld = dl2ld(dl)
    
    l = ld[0]['logits_imgs'].shape[0]
    # print(ld[0].keys())
    # dict_keys(['img_key', 'label_imgs', 'densepose', 'reference_img', 'reference_img_controlnet', 'reference_img_vae', 'background_mask', 'logits_imgs'])
    # print(ld[0]['logits_imgs'].shape)
    # torch.Size([3, 256, 192])
    # print('outputslogits_imgs',outputs['logits_imgs'].shape)
    # outputslogits_imgs torch.Size([10, 3, 256, 192])
    frames = outputs['logits_imgs']
    # frames = rearrange(frames, "f c h w -> f h w c")
    
    # save_vd_dir = "/home/nfs/jsh/DisCo"
    # save_vd_path = os.path.join(eval_save_filename, 'movie.mp4')
    # vd_outputs = []
    for i, x in enumerate(frames):
        # print('xxxx',x.shape)
        # torch.Size([256, 192, 3])
        x = tensor2pil(x)[0]
        # x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        # x = (x * 255).numpy().astype(np.uint8)
        save_dir = os.path.join(eval_save_filename, 'pred_image')
        os.makedirs(save_dir, exist_ok=True)
        x.save(os.path.join(save_dir, f'0{i}.png'))
        # vd_outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    # imageio.mimsave(save_vd_path, vd_outputs, fps=4)

    # image = tensor2pil(sample['logits_imgs'])[0]
    # print('!!!',os.path.join(pred_save_path, prefix + postfix + '.png'))
    # image.save(os.path.join(pred_save_path, prefix + postfix + '.png'))
    '''
    for _, sample in enumerate(ld):
        _name = 'nuwa'
        if 'input_text' in sample:
            _name = sample['input_text'][:200]
        if 'img_key' in sample:
            _name = sample['img_key']
        char_remov = [os.sep, '.', '/']
        for c in char_remov:
            _name = _name.replace(c, '')
        prefix = "_".join(_name.split(" "))
        # prefix += splice
        # postfix = '_' + str(round(time.time() * 10))
        postfix = ''

        if 'label_imgs' in sample:
            image = sample['label_imgs']
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(image)[0]
            try:
                image.save(os.path.join(gt_save_path, prefix + postfix + '.png'))
            except Exception as e:
                print(f'some errors happened in saving label_imgs: {e}')
        if 'logits_imgs' in sample:
            # print('samplelogits_imgs',sample['logits_imgs'].shape)
            # torch.Size([3, 256, 192])
            image = tensor2pil(sample['logits_imgs'])[0]
            try:
                print('!!!',os.path.join(pred_save_path, prefix + postfix + '.png'))
                image.save(os.path.join(pred_save_path, prefix + postfix + '.png'))
            except Exception as e:
                print(f'some errors happened in saving logits_imgs: {e}')
        if 'cond_imgs' in sample and sample['cond_imgs'] is not None: # pose
            image = tensor2pil(sample['cond_imgs'])[0]
            try:
                image.save(os.path.join(cond_save_path, prefix + postfix + '.png'))
            except Exception as e:
                print(f'some errors happened in saving logits_imgs: {e}')
        if 'reference_img' in sample:
            image = sample['reference_img']
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(image)[0]
            try:
                image.save(os.path.join(ref_save_path, prefix + postfix + '.png'))
            except Exception as e:
                print(f'some errors happened in saving label_imgs: {e}')
        if 'reference_img_controlnet' in sample:
            image = sample['reference_img_controlnet']
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(image)[0]
            try:
                image.save(os.path.join(ref_control_save_path, prefix + postfix + '.png'))
            except Exception as e:
                print(f'some errors happened in saving label_imgs: {e}')
    '''
    return gt_save_path, pred_save_path

def tensor2pil(images):
    # c, h, w
    images = images.cpu().permute(1, 2, 0).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images