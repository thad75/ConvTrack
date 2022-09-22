# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
positional encodings for the transformer.
"""
import math
import torch
import torch.nn as nn
from util.misc import NestedTensor
from typing import List
from torchvision.transforms.functional import crop as Crop
from util.misc import NestedTensor
import torchvision.transforms as T
from einops import rearrange, repeat

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) - 0.5
        x_embed = not_mask.cumsum(2, dtype=torch.float32) - 0.5
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps + 0.5) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps + 0.5) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


@torch.no_grad()


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


# def gen_sineembed_for_position(pos_tensor):
#     scale = 2 * math.pi
#     dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
#     dim_t = 10000 ** (2 * (dim_t // 2) / 128)
#     x_embed = pos_tensor[:, :, 0] * scale
#     y_embed = pos_tensor[:, :, 1] * scale
#     pos_x = x_embed[:, :, None] / dim_t
#     pos_y = y_embed[:, :, None] / dim_t
#     pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
#     pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
#     pos = torch.cat((pos_y, pos_x), dim=2)
#     return pos


class Cropper(nn.Module):
    # TODO : could be lots of overlapping crop
    
    def __init__(self, num_queries, crop_size = 100, hidden_dim = 256, projection = None):
        super().__init__()
        self.crop_size = crop_size
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
#         self.crop = Crop(crop_size)
        
    def forward(self,samples: NestedTensor,  targets:List=None):
        list_of_crop = []
        images = samples.decompose()[0]
        BS, C , H, W = images.shape
        for i,target in enumerate(targets):
            if 'boxes' in target:
                boxes = target["boxes"]
                if len(boxes): 
                    for bbox in boxes :  
                        x,y,w,h= bbox
                        x,y,w,h = x, y  ,w,h                
                        x,y,w,h = x*W,y*H,w*W,h*H
                        cropy = Crop(images[i,...],x.int(),y.int(),w.int(),h.int())   
                        cropy = T.Resize((self.crop_size,self.crop_size))(cropy)
                        list_of_crop.append(cropy)
        try : 
            crop = nested_tensor_from_tensor_list(list_of_crop)#torch.stack(list_of_crop, dim = 0)
            return crop
        except : 
            return None

def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, learnedwh=None):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
   # print(memory.shape)
#     memory = rearrange(memory, 'hw bs hidden -> bs hw hidden')
    
    #spatial_shapes = rearrange(spatial_shapes, 'bs lvl -> ')
   # print('Inside Gen', memory.shape, memory_padding_mask.shape,spatial_shapes.shape)

    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
     #   print(lvl, H_,W_, N_,memory_padding_mask[:, _cur:(_cur + H_ * W_)].shape)
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        # import ipdb; ipdb.set_trace()

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            # import ipdb; ipdb.set_trace()
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    # import ipdb; ipdb.set_trace()
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    # output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    # output_memory = output_memory.masked_fill(~output_proposals_valid, float('inf'))

    return output_memory, output_proposals

    
def build_position_encoding(position_embedding = 'sine', hidden_dim = 256):
    if position_embedding in ('sine'):
        position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    else:
        raise ValueError(f"Unknown args.position_embedding: {position_embedding}.")
    return position_embedding
