# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
from json import detect_encoding
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50
from .positional import PositionalEncoding2D
from einops import rearrange
from torch.nn import MultiheadAttention
import torchvision
from models.misc import _get_clones, MLP
from models.criterion import *
from models.tracklet import Tracklet
from models.criterion import build_criterion
from models.transformer import * 
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, reduce_dict)
from typing import List
import numpy as np
# from criterion import build_criterion
class Model(pl.LightningModule):
    """ This is a PL Model """
    def __init__(self, 
                backbone,
                transformer,
                num_queries,
                tracker, 
                memory_bank,
                num_classes = 1,
                num_encoder_stack=6,
                num_decoder_stack = 6,
                matcher= None,
                hidden_dim = 256 , 
                embed_dim = 128,
                multiscale = True, 
                masks = True, 
                weights = None):
        """ Initializes the model.
        Parameters:
           
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_encoder_stack = num_encoder_stack
        self.num_decoder_stack = num_decoder_stack
        self.multiscale = multiscale
        self.num_feature_levels = 4 if self.multiscale else 1          #
        self.hidden_dim = hidden_dim
        self.aux_loss = True
        # Init Backbone, Trasnformer, Matcher, Weights
        # TODO : Change the BackBone with DETR BackBine
        self.backbone = backbone        
        self.transformer = transformer

        self.matcher = matcher
        self.weights = weights
        self.tracker = tracker
        self.criterion_train= build_criterion(matcher , 
                                              weights,  
                                              aux_loss = self.aux_loss, 
                                              num_classes= self.num_classes,
                                              dec_layers = self.num_decoder_stack,
                                              masks = masks, 
                                              type = 'Other')

        self.criterion_test= build_criterion(matcher , 
                                              weights,  
                                              aux_loss = False, 
                                              num_classes= self.num_classes,
                                              dec_layers = self.num_decoder_stack,
                                              masks = masks, 
                                              type = 'Other')

        if self.multiscale:
            input_proj_list = []
            i=0
            for _ in range(self.num_feature_levels):
                i+=1
                print('Created', i)
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                                                     nn.GroupNorm(32, self.hidden_dim),))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # TODO : Retrieve the n_epoch from the Script.py
            if self.current_epoch >= 25:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),)])
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                                  nn.GroupNorm(32, self.hidden_dim),)])
        self.masks = masks 
        # Init Loss Stuff
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        
        self.upsample_layer = nn.Sequential(nn.PixelShuffle(2),
                                            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1))


        # 3D-2D Conv
        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=2*hidden_dim, 
                                            out_channels=hidden_dim, 
                                            kernel_size=2, 
                                            groups=hidden_dim), 
                                nn.GELU())


        self.temporal_conv = nn.Sequential(nn.Conv3d(in_channels=hidden_dim, 
                                             out_channels=hidden_dim, 
                                             kernel_size=2),
                                nn.ReLU())
        # Heads
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        self.query_embed = nn.Embedding(self.num_queries,4*self.hidden_dim) # The Temproal Features are in shape 2*hidden_dim 

        # HyperParameter 
        # For Transxformers          
        self.transformer.enc_out_bbox_embed = self.bbox_embed
        self.transformer.enc_out_class_embed = self.class_embed
        
        self.class_embed = _get_clones(self.class_embed, num_decoder_stack)
        self.bbox_embed = _get_clones(self.bbox_embed, num_decoder_stack)
        
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.loss_train = ...
        self.loss_val = ...
        
        # ROI Porjectort
        self.roi_projector = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim // 4, kernel_size=1, stride=1, padding=0),
                                            nn.ReLU())
        
        self.roi_projector_2 = nn.Sequential(nn.Linear(hidden_dim// 4 * 7 * 7,hidden_dim*2),
                                            nn.ReLU())
                        
        # On MultiScale Data
        # Other Stuff
        self.memory_bank = memory_bank
        
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def with_pos_embed(self,x, pos_x= None):
        if pos_x is not None:
            return x + pos_x
        else : 
            return x
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coords, outputs_embedding):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_embedding': c}
                for a, b,c in zip(outputs_class[:-1], outputs_coords[:-1], outputs_embedding[:-1])]
    
    
    def convert_to_nested_tensor_from_list(self,samples):
        if isinstance(samples, (list, np.array)):
            samples = [torch.from_numpy(image) for image in samples]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        return samples
    
    def backbone_and_projection(self,samples):
        
        features, pos_embeds = self.backbone(samples)
        srcs,masks= [],[]
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos_embeds.append(pos_l) 
        return srcs,pos_embeds, masks
    
    
############################ Tracking Method #################################    
    def init_track(self):
        nquery, dim = self.num_queries, 2*self.hidden_dim
        tracklets = Tracklet() # Define tracklet
        
        # defining all stuff needed for a tracklets
        tracklets.ref_points = self.transformer.reference_points(self.query_embed.weight[:, :dim // 2])
        tracklets.query_pos = self.query_embed.weight
        tracklets.output_embedding = torch.zeros((self.num_queries, dim >> 1), device=self.device)
        tracklets.obj_idxes = torch.full((len(tracklets),), -1, dtype=torch.long, device=self.device)        
        tracklets.matched_gt_idxes = torch.full((len(tracklets),), -1, dtype=torch.long, device=self.device)
        tracklets.disappear_time = torch.zeros((len(tracklets), ), dtype=torch.long, device=self.device)
        tracklets.iou = torch.zeros((len(tracklets),), dtype=torch.float, device=self.device)
        tracklets.scores = torch.zeros((len(tracklets),), dtype=torch.float, device=self.device)
        tracklets.track_scores = torch.zeros((len(tracklets),), dtype=torch.float, device=self.device)
        tracklets.pred_boxes = torch.zeros((len(tracklets), 4), dtype=torch.float, device=self.device)
        tracklets.pred_logits = torch.zeros((len(tracklets), self.num_classes), dtype=torch.float, device=self.device)
        # init the memory bank 
        mem_bank_len = self.mem_bank_len  # TODO : Define this thing
        tracklets.mem_bank = torch.zeros((len(tracklets), mem_bank_len, dim // 2), dtype=torch.float32, device=self.device)
        tracklets.mem_padding_mask = torch.ones((len(tracklets), mem_bank_len), dtype=torch.bool, device=self.device)
        tracklets.save_period = torch.zeros((len(tracklets), ), dtype=torch.float32, device=self.device)
        
        return tracklets.to(self.device)
        
############################ END Tracking Method #################################    
    
    def pad_roi(self, tensor, h,w ):
        zeros = torch.tensor([0,0,0,0], device = self.device)
        h_w = torch.tensor([w,h,w,h], device = self.device)
        if tensor.shape[0]< self.num_queries:
            add_zero = self.num_queries - tensor.shape[0]
            zeros = zeros.repeat(add_zero).view(-1,4)
            tensor = torch.cat((torch.Tensor(tensor).to( self.device), zeros), dim = 0)
        return tensor*h_w
    
    def target_to_tracklets(self,samples: NestedTensor):
        gt_instances = Tracklet()
        gt_instances.boxes = samples['boxes']
        gt_instances.labels = samples['labels']
        gt_instances.obj_ids = samples['obj_ids']
        gt_instances.area = samples['area']
        
        
        
        return gt_instances
        
        
        
    def forward_pre(self, samples: NestedTensor, targets:List=None):
        samples_ref = samples['image']
        samples_ref = self.convert_to_nested_tensor_from_list(samples_ref).to(self.device)
        srcs_ref,pos_embeds_ref, masks_ref = self.backbone_and_projection(samples_ref)
        for ref in zip(srcs_ref):

            temporal_src = self.temporal_conv(temporal_src)
            spatial_src = self.spatial_conv(spatial_src)
            sp_src = torch.cat([temporal_src.view(spatial_src.shape), spatial_src], dim = 1)
            srcs.append(sp_src)
            pos_embeds.append(PositionalEncoding2D(sp_src.shape[1])(sp_src))
            b,c,h,w = sp_src.shape
            masks.append(torch.ones((b,h,w), dtype=torch.bool, device = self.device))
    
    
    def forward(self, samples: NestedTensor, targets= None):
        print(samples.keys())
        print(samples['boxes'])
        print()
        print(samples['meta'])
        
        gt_instances = self.target_to_tracklets(samples)
        if self.training:
            self.criterion_train.initialize_for_single_clip(samples['gt_instances'])
        tracklets = self.init_track()
        ret = self.forward_loop(samples, tracklets)
        return ret
        
    
    def forward_loop(self, samples: NestedTensor,  tracklets= None):
        """
        ref boxes should be a list of Tensor of [1,4], with (x1,y1,x2,y2)
        
        """
        # Preprocesing the Samples        
        samples_ref = samples['image']
        samples_tgt = samples['pre_img']
        bs = len(samples_ref)
        c,h,w = samples_ref[0].shape
        samples_ref = self.convert_to_nested_tensor_from_list(samples_ref).to(self.device)
        samples_tgt = self.convert_to_nested_tensor_from_list(samples_tgt).to(self.device)        
        srcs_tgt ,pos_embeds_tgt , masks_tgt = self.backbone_and_projection(samples_tgt)
        srcs_ref,pos_embeds_ref, masks_ref = self.backbone_and_projection(samples_ref)

        # Local Attention : Here we introduce the new module with conv3d and conv2d
        # Creating Temporal Features
        assert len(srcs_tgt)== len(srcs_ref)
        srcs, pos_embeds, masks = [], [], []
        for tgt, ref in zip(srcs_tgt,srcs_ref):
            temporal_src = torch.stack([tgt,ref], dim = 2)
            spatial_src = rearrange(temporal_src.clone(), 'bs c t h w -> bs (c t ) h w')
            temporal_src = self.temporal_conv(temporal_src)
            spatial_src = self.spatial_conv(spatial_src)
            sp_src = torch.cat([temporal_src.view(spatial_src.shape), spatial_src], dim = 1)
            srcs.append(sp_src)
            pos_embeds.append(PositionalEncoding2D(sp_src.shape[1])(sp_src))
            b,c,h,w = sp_src.shape
            masks.append(torch.ones((b,h,w), dtype=torch.bool, device = self.device))            
        print('###################### Features Stuff ##########################')
  
        # Extracting ROI Features from the feature map
        roi_bbox = [self.pad_roi(i,h,w) for i in samples['init_bbox']]
        roi_features =  torchvision.ops.roi_align(srcs_ref[-1], 
                                                  roi_bbox,
                                                  output_size = (7,7),
                                                  spatial_scale = 1.0,
                                                  aligned= True)
        # Projecting the ROI Features
        roi_features =   self.roi_projector(roi_features)
        roi_features = roi_features.reshape(bs*self.num_queries, -1)
        roi_features =  self.roi_projector_2(roi_features)
        roi_features = roi_features.reshape(bs,self.num_queries, -1)


        # Let's go Transformer
        if self.query_embed : 
            query_embed = self.query_embed.weight 
        else :
            query_embed = None
        hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact= self.transformer(srcs = srcs,
                             masks = masks,
                             pos_embeds = pos_embeds,
                             query_embed = query_embed,
                             ref_pts =tracklets.ref_points, # Check why None 
                             roi_features = roi_features)
        outputs_classes = []
        outputs_coords = []
        outputs_embedding = []
        
        for lvl in range(len(hs)):           
            reference = init_reference_out if lvl == 0 else inter_references_out[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            # print(outputs_class)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_embedding.append(hs[lvl])
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_embedding = torch.stack(outputs_embedding)
        ref_pts_all = torch.cat([init_reference_out[None],inter_references_out[:, :, :, :2]], dim=0)
        # Send to Single Object Head and use that as teacher for student network ead
        
        out = {'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1], 
               'pred_embedding': outputs_embedding[-1]}
        #
        print('Trasnformer OK')
        if self.aux_loss:
#             print('Into Auxiliary Loss')
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coords, outputs_embedding)
    
        # Stuff on Tracking : Update and Init
        if self.training:
            track_scores = outputs_class[-1, 0, :].sigmoid().max(dim=-1).values
        else:
            track_scores = outputs_class[-1, 0, :, 0].sigmoid()            
        
        if tracklets is not None: 
            tracklets.scores = track_scores
            tracklets.pred_logits = outputs_class[-1, 0]
            tracklets.pred_boxes = outputs_coord[-1, 0]
            tracklets.output_embedding = hs[-1, 0]
        if self.training and tracklets is not None:
            # the track id will be assigned by the mather.
            out['tracklets'] = tracklets
            tracklets = self.criterion_train.match_for_single_frame(out)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(tracklets)
        if self.memory_bank is not None and tracklets is not None:
            tracklets = self.memory_bank(tracklets)
            if self.training:
                self.criterion_train.calc_loss_for_track_scores(tracklets)
        tmp = {}
        tmp['init_tracklets'] = self._generate_empty_tracks()
        tmp['tracklets'] = tracklets
        out_tracklets = self.track_embed(tmp)
        out['tracklets'] = out_tracklets
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4,weight_decay= 1e-4) 
        scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 10)   
        return {"optimizer":optimizer, "lr_scheduler":scheduler}

    def training_step(self,batch, batch_idx):
        print(batch.keys())
        print(batch['boxes'][0].shape)
        # print(batch['init_boxes'])
        # print(batch['meta'])
        tracklets = self.init_track()
        
        out = self.forward(batch, tracklets)
        loss = ...
        return loss
        
        
    def validation_step(self,batch,  batch_idx):
        print(batch.keys())

        data, label = batch
        
        out = self.forward(data)
        loss = ...
        return loss
        
        
    def on_validation_start(self):
        ...

        
    def on_validation_end(self):
        ...

# data = torch.rand(2,3,270,480)
# norm_tensor = torch.Tensor((1/270,1/480,1/270,1/480))

# print(torch.Tensor((0,15,50,80))*norm_tensor)
# ref_boxes = [torch.Tensor((0,15,50,80))*norm_tensor,torch.Tensor((15,90,59,100))*norm_tensor]
# model = Model()

# forward = model(data,data,ref_boxes)

# class TrackerPostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api"""
#     def __init__(self):
#         super().__init__()

#     @torch.no_grad()
#     def forward(self, tracklets: Instances, target_size) -> Instances:
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation)
#                           For visualization, this should be the image size after data augment, but before padding
#         """
#         out_logits = tracklets.pred_logits
#         out_bbox = tracklets.pred_boxes

#         prob = out_logits.sigmoid()
#         # prob = out_logits[...,:1].sigmoid()
#         scores, labels = prob.max(-1)

#         # convert to [x0, y0, x1, y1] format
#         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
#         # and from relative [0, 1] to absolute [0, height] coordinates
#         img_h, img_w = target_size
#         scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
#         boxes = boxes * scale_fct[None, :]

#         tracklets.boxes = boxes
#         tracklets.scores = scores
#         tracklets.labels = labels
#         tracklets.remove('pred_logits')
#         tracklets.remove('pred_boxes')
#         return tracklets