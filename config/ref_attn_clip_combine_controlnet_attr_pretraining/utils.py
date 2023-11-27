
import torch
import torch.nn.functional as F
from einops import rearrange
# from ..modules.utils import back_projection, get_x_2d

from torch import nn, einsum

class PosEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(PosEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        # self.funcs = [torch.sin, torch.cos]
        # self.out_channels = in_channels*(len(self.funcs)*N_freqs)
        if N_freqs <= 80:
            base = 2
        else:
            base = 5000**(1/(N_freqs/2.5))
        if logscale:
            freq_bands = base**torch.linspace(0,
                                              N_freqs-1, N_freqs).cuda()[None, None]
        else:
            freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs).cuda()
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        shape = x.shape[:-1]
        x = x.reshape(-1, 2, 1)
        # print('device',x.device)
        # print('device',self.freq_bands.device)
        encodings = x * self.freq_bands
        sin_encodings = torch.sin(encodings)  # (n, c, num_encoding_functions)
        cos_encodings = torch.cos(encodings)
        pe = torch.cat([sin_encodings, cos_encodings], dim=1)
        pe = pe.reshape(*shape, -1)
        return pe


def get_correspondence(depth, pose, K, x_2d):
    b, h, w = depth.shape
    x3d = back_projection(depth, pose, K, x_2d)
    x3d = rearrange(x3d, 'b h w c -> b c (h w)')
    x3d = K[:, :3, :3]@x3d
    x3d = rearrange(x3d, 'b c (h w) -> b h w c', h=h, w=w)
    x2d = x3d[..., :2]/(x3d[..., 2:3]+1e-6)

    mask = depth == 0
    x2d[mask] = -1000000
    x3d[mask] = -1000000

    return x2d, x3d

def get_key_value(key_value, xy_l, xy_r, depth_query, depths, imap_query, imap, ori_h, ori_w, ori_h_r, ori_w_r, query_h, query_w):

    b, c, h, w = key_value.shape
    # torch.Size([1, 320, 64, 48])
    query_scale = ori_h//query_h
    # 16
    # print(query_scale)
    key_scale = ori_h_r//h
    # 16
    # print('xy_l',xy_l.shape)
    # xy_l torch.Size([1,1024, 768, 2])
    xy_l = xy_l[:, query_scale//2::query_scale,query_scale//2::query_scale]/key_scale-0.5
    # print('xy_l',xy_l.shape)
    # xy_l torch.Size([1, 64, 48, 2])
    # print('key_value',key_value.shape)
    # key_value torch.Size([1, 320, 64, 48])
    # print('depth_query',depth_query.shape)
    # depth_query torch.Size([1, 1024, 768])
    # print('depths',depths.shape)
    # depths torch.Size([1,1024, 768])

    # torch.Size([1, 64, 48,2])
    key_values = []

    xy_proj = []
    depth_proj = []
    mask_proj = []
    kernal_size = 1
    # depth_query # torch.Size([1, 512, 384])
    # depths torch.Size([1, 512, 384])
    depth_query = depth_query[:, query_scale//2::query_scale,query_scale//2::query_scale]
    imap = imap[:, query_scale//2::query_scale,query_scale//2::query_scale]
    imap_query = imap_query[:, query_scale//2::query_scale,query_scale//2::query_scale]
    # depth_query torch.Size([1, 64, 48])
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone()
            xy_l_norm[..., 0]=xy_l[..., 1]
            xy_l_norm[..., 1]=xy_l[..., 0]
            # [1, 64, 48, 2]
            # displacement
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j
            xy_l_rescale = (xy_l_norm+0.5)*key_scale
            xy_l_round = xy_l_rescale.round().long()
            mask = (xy_l_round[..., 0] >= 0)*(xy_l_round[..., 0] < ori_w) * (
                xy_l_round[..., 1] >= 0)*(xy_l_round[..., 1] < ori_h)
            # print(imap.shape)
            mask_head = imap>22.5
            mask_bg = imap<0.5
            # print('1',torch.sum(mask!=0)/mask.numel())
            xy_l_round[..., 0] = torch.clamp(xy_l_round[..., 0], 0, ori_w-1)
            xy_l_round[..., 1] = torch.clamp(xy_l_round[..., 1], 0, ori_h-1)
            # xy_l_round [1, 64, 48,2]
            # xy_l_round[b_i, ..., 1] ([64, 48])
            # depths torch.Size([1, 64, 48])
            # depths[b_i, xy_l_round[b_i, ..., 1], xy_l_round[b_i, ..., 0]
            # torch.Size([64, 48])
            depth_i = torch.stack([depths[b_i, xy_l_round[b_i, ..., 1], xy_l_round[b_i, ..., 0]]
                                  for b_i in range(b)])
            # from IPython import embed; embed()
            # depth_i torch.Size([1(b),64, 48])
            # print('in,depth_i',depth_i.shape)
            # in,depth_i torch.Size([1, 64, 48])
            
            mask = mask*(depth_i > 0)
            
            # print('2',torch.sum(mask!=0)/mask.numel())
            # print('in,mask',mask.shape)
            # print('===================',mask)
            # in,mask torch.Size([1, 64, 48])
            depth_i[~mask] = 100
            depth_proj.append(depth_i)
            
            # mask = mask*(depth_query>0)
            
            mask = mask+mask_head
            
            # print('3',torch.sum(mask!=0)/mask.numel())
            mask_proj.append(mask)

            xy_proj.append(xy_l_rescale.clone())

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            # print('===================',xy_l_norm)
            xy_l_norm = xy_l_norm.to(dtype=key_value.dtype)
            _key_value = F.grid_sample(key_value, xy_l_norm, align_corners=True)
            # _key_value = F.grid_sample(key_value, xy_l_norm, padding_mode="zeros",align_corners=True)
            # from IPython import embed; embed()
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    # print('xy_proj',xy_proj.shape)
    # xy_proj torch.Size([1, 1, 64, 48, 2])
    # torch.Size([1(b),1,64, 48,2])
    depth_proj = torch.stack(depth_proj, dim=1)
    # print('depth_proj',depth_proj.shape)
    # depth_proj torch.Size([1, 1, 64, 48])
    # torch.Size([1(b),1,64, 48])
    mask_proj = torch.stack(mask_proj, dim=1)
    # print('mask_proj',mask_proj.shape)
    # mask_proj torch.Size([1, 1, 64, 48])
    # torch.Size([1(b),1,64, 48])
    xy_proj = rearrange(xy_proj, 'b n h w c -> (b n) h w c')
    depth_proj = rearrange(depth_proj, 'b n h w -> (b n) h w')


    # xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    # xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    
    # xy = torch.tensor(xy, device=key_value.device).float()[
    #     None].repeat(xy_proj.shape[0], 1, 1, 1)   
    # print('-depth_query',depth_query)
    xy_rel = (depth_query-depth_proj).abs()[...,None] # depth check

    xy_rel = rearrange(xy_rel, '(b n) h w c -> b n h w c', b=b)

    key_values = torch.stack(key_values, dim=1)
    # 1 1 320 64 48
    # 1 1 64 48 2
    # 1 1 64 48
    # print('key_values',key_values.shape)
    # print('xy_rel',xy_rel.shape)
    # print('--------------',xy_rel)
    # print('mask_proj',mask_proj.shape)
    # print('--------------',mask_proj)

    # key_values torch.Size([1, 1, 320, 64, 48])
    # xy_rel torch.Size([1, 1, 64, 48, 1])
    # mask_proj torch.Size([1, 1, 64, 48])
    return key_values, xy_rel, mask_proj

# query, key_value, key_value_xy, mask = get_query_value(
#                     x_left, x_right, xy_l, xy_r, depth_query, _depths, img_h, img_w, img_h, img_w)
# 
def get_query_value(query, key_value, xy_l, xy_r, depth_query, depths, imap_query, imap, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    b = query.shape[0]
    m = key_value.shape[1]
    # m=1
    # torch.Size([1, 1, 320, 64, 48])
    key_values = []
    masks = []
    xys = []
    # guess
    # xy_l torch.Size([1, 1, 512, 384,2])
    # xy_r torch.Size([1, 1, 512, 384,2])
    # print('get_query_value:xy_l',xy_l.shape)
    # get_query_value:xy_l torch.Size([1, 1, 1024, 768, 2])
    # depth_query 1 512 384
    # depths  1 1 512 384
    for i in range(m):
        _, _, q_h, q_w = query.shape
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], xy_r[:, i], depth_query, depths[:, i],imap_query,imap[:, i],
                                               img_h_l, img_w_l, img_h_r, img_w_r, q_h, q_w)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    # 1 1(m) 320 64 48
    # print('key_value',key_value.shape)
    # key_value torch.Size([1, 1, 320, 64, 48])
    xy = torch.cat(xys, dim=1)
    # print('xy',xy.shape)
    # xy torch.Size([1, 1, 64, 48, 1])
    # 1 1(m) 64 48 2
    mask = torch.cat(masks, dim=1)
    # print('mask',mask.shape)
    # mask torch.Size([1, 1, 64, 48])
    # 1 1(m) 64 48
    return query, key_value, xy, mask