
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import Optional, List
import math

from models.transformer_encoder import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from models.transformer_decoder import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
from .ops.modules import MSDeformAttn
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_



class DeformableTransformer(nn.Module):
    def __init__(self, 
                d_model=256, 
                nhead=8,
                num_encoder_layers=6, 
                num_decoder_layers=6, 
                dim_feedforward=1024, 
                dropout=0.1,
                activation="relu", 
                return_intermediate_dec=False,
                num_feature_levels=4, 
                dec_n_points=4,  
                enc_n_points=4,     
                two_stage=False, 
                two_stage_num_proposals=300, 
                decoder_self_cross=True, 
                sigmoid_attn=False,
                extra_track_attn=False):

        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, 
                                                        dim_feedforward,
                                                        dropout,
                                                        activation,
                                                        num_feature_levels, 
                                                        nhead, 
                                                        enc_n_points,
                                                        sigmoid_attn=sigmoid_attn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, 
                                                        dim_feedforward,
                                                        dropout, 
                                                        activation,
                                                        num_feature_levels,
                                                        nhead, 
                                                        dec_n_points,
                                                        decoder_self_cross,
                                                        sigmoid_attn=sigmoid_attn, 
                                                        extra_track_attn=extra_track_attn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, ref_pts=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            
            if ref_pts is None:
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
            init_reference_out = reference_points
        # decoder
        hs, inter_references = self.decoder(tgt, 
                                            reference_points, 
                                            memory,
                                            spatial_shapes, 
                                            level_start_index, 
                                            valid_ratios, 
                                            query_embed, 
                                            mask_flatten)

        inter_references_out = inter_references

        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


def build_transformer(hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6,     
                        dim_feedforward=1024, dropout=0.1, activation="relu", 
                        return_intermediate_dec=False, num_feature_levels=4, 
                        dec_n_points=4, enc_n_points=4, two_stage=False, two_stage_num_proposals=300, 
                        decoder_self_cross=True, sigmoid_attn=False,extra_track_attn=False):
    """
    It returns a DeformableTransformer object with the following parameters:
    
    hidden_dim=256, 
    nheads=8,
    num_encoder_layers=6, 
    num_decoder_layers=6, 
    dim_feedforward=1024, 
    dropout=0.1,
    activation="relu", 
    return_intermediate_dec=False,
    num_feature_levels=4, 
    dec_n_points=4,  
    enc_n_points=4,     
    two_stage=False, 
    two_stage_num_proposals=300, 
    decoder_self_cross=True, 
    sigmoid_attn=False,
    extra_track_attn=False
    
    :param hidden_dim: the dimension of the hidden layer in the transformer, defaults to 256 (optional)
    :param nheads: number of attention heads, defaults to 8 (optional)
    :param num_encoder_layers: number of encoder layers, defaults to 6 (optional)
    :param num_decoder_layers: number of decoder layers, defaults to 6 (optional)
    :param dim_feedforward: The dimension of the "feedforward" layer in each encoder and decoder block,
    defaults to 1024 (optional)
    :param dropout: dropout rate
    :param activation: The activation function to use, defaults to relu (optional)
    :param return_intermediate_dec: whether to return intermediate decoder outputs, defaults to False
    (optional)
    :param num_feature_levels: number of feature levels in the encoder and decoder, defaults to 4
    (optional)
    :param dec_n_points: number of points in the decoder, defaults to 4 (optional)
    :param enc_n_points: number of points in the encoder, defaults to 4 (optional)
    :param two_stage: whether to use the two-stage model, defaults to False (optional)
    :param two_stage_num_proposals: number of proposals to use in the second stage of the two-stage
    model, defaults to 300 (optional)
    :param decoder_self_cross: whether to use cross-attention between the decoder and itself, defaults
    to True (optional)
    :param sigmoid_attn: whether to use sigmoid attention or not, defaults to False (optional)
    :param extra_track_attn: whether to use the extra track attention, defaults to False (optional)
    :return: A DeformableTransformer object
    """


#     print(num_queries, smca)
    return DeformableTransformer(d_model=hidden_dim, 
                                nhead=nheads,
                                num_encoder_layers=num_encoder_layers, 
                                num_decoder_layers=num_decoder_layers, 
                                dim_feedforward=dim_feedforward, 
                                dropout=dropout,
                                activation=activation, 
                                return_intermediate_dec=return_intermediate_dec,
                                num_feature_levels=num_feature_levels, 
                                dec_n_points=dec_n_points,  
                                enc_n_points=enc_n_points,     
                                two_stage=two_stage, 
                                two_stage_num_proposals=two_stage_num_proposals, 
                                decoder_self_cross=decoder_self_cross, 
                                sigmoid_attn=sigmoid_attn,
                                extra_track_attn=extra_track_attn)

build_transformer()