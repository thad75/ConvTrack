import torch
import torch.nn as nn
from .misc import _get_clones, _get_activation_fn
from models.ops.modules import MSDeformAttn
from util.misc import inverse_sigmoid

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1, 
                 activation="relu",
                 n_levels=4, 
                 n_heads=8, 
                 n_points=4, 
                 sigmoid_attn=False):
        super().__init__()

        # self attention
        
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src = self.with_pos_embed(src, pos)
        src2 = self.self_attn(query = src, 
                               reference_points = reference_points, 
                               input_flatten = src, 
                               input_spatial_shapes = spatial_shapes, 
                               input_level_start_index = level_start_index, 
                               input_padding_mask = padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output