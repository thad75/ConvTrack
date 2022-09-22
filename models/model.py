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
from models.transformer import * 
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, reduce_dict)
from typing import List
import numpy as np
class Model(pl.LightningModule):
    """ This is a PL Model """
    def __init__(self, 
                backbone,
                transformer,
                num_queries,
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
        self.nm_queries = num_queries
        self.num_classes = num_classes
        self.num_encoder_stack = num_encoder_stack
        self.num_decoder_stack = num_decoder_stack
        self.multiscale = multiscale
        self.num_feature_levels = 4 if self.multiscale else 1          #
        self.hidden_dim = hidden_dim

        self.weights = weights
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
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # TODO : Change the BackBone with DETR BackBine
        self.backbone = backbone
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.upsample_layer = nn.Sequential(nn.PixelShuffle(2),
                                            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1))

        self.tokenizer = ...

        self.heads = ...

        self.head_detect_sot = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=1, stride=1, padding=0)
        self.head_class_sot = nn.Conv2d(in_channels=1,out_channels=2,kernel_size=1, stride=1, padding=0)
        self.head_mask_sot = nn.Conv2d(in_channels=1,out_channels=2,kernel_size=1, stride=1, padding=0) 

        self.head_detect_mot = ...
        self.head_class_mot = ...
        self.head_mask_mot = ...
        
        self.backbone = backbone
        self.transformer = transformer
        self.criterion = ...
        self.matcher = matcher
        # HyperParameter 
        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=2*hidden_dim, out_channels=hidden_dim, kernel_size=2, groups=hidden_dim), 
                                        nn.GELU())

        
        self.temporal_conv = nn.Sequential(nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=2),
                                        nn.ReLU())

        self.loss_train = ...
        self.loss_val = ...

        # On MultiScale Data
       

    def with_pos_embed(self,x, pos_x= None):
        if pos_x is not None:
            return x + pos_x
        else : 
            return x

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
    
    
    
    def init_track(self):
        ...
    
    
    def forward(self, samples: NestedTensor,  targets:List=None):
        """
        ref boxes should be a list of Tensor of [1,4], with (x1,y1,x2,y2)
        
        """
        # Preprocesing the Samples
        samples_ref = samples['image']
        samples_tgt = samples['pre_img']
#         print('device, ',samples_ref[0].device)
#         print(self.backbone.device)
        samples_ref = self.convert_to_nested_tensor_from_list(samples_ref).to(self.device)
        samples_tgt = self.convert_to_nested_tensor_from_list(samples_tgt).to(self.device)
        
        features, pos_embeds = self.backbone(samples_ref)
        srcs_tgt ,pos_embeds_tgt , masks_tgt = self.backbone_and_projection(samples_tgt)
        srcs_ref,pos_embeds_ref, masks_ref = self.backbone_and_projection(samples_ref)

        # at this step we need for the transfomer : srcs , masks, pos, tracking instance ? , reference poitns (track_isntance_ref points)
        # Local Attention : Here we introduce the new module with conv3d and conv2d
        assert len(srcs_tgt)== len(srcs_ref)
        # TODO : Verify if this mask confinguration is ok
        srcs, pos_embeds, masks = [], [], []
        for tgt, ref in zip(srcs_tgt,srcs_ref):
            temporal_src = torch.stack([tgt,ref], dim = 2)
            spatial_src = rearrange(temporal_src.clone(), 'bs c t h w -> bs (c t ) h w')
            temporal_src = self.temporal_conv(temporal_src)
            spatial_src = self.spatial_conv(spatial_src)
            sp_src = torch.cat([temporal_src.view(spatial_src.shape), spatial_src], dim = 1)
            srcs.append(sp_src)
            pos_embeds.append(PositionalEncoding2D(sp_src.shape[1])(sp_src))
            masks.append(torch.ones(sp_src.shape, device = self.device))
            
        print('###################### Features Stuff ##########################')
        print([i.shape for i in srcs], [i.shape for i in pos_embeds])

        # TODO : Let's go into the transforemr  

        hs= self.transformer(srcs = srcs,
                             masks = masks,
                             pos_embeds = pos_embeds,
                             query_embed = query_embed,
                             ref_pts =... ,
                             valid_ratio= ...)



        f_spatio_temporal = self.attention(f_spatio_temporal,f_spatio_temporal,f_spatio_temporal)[0]

        bs, c, h , w  = f_target.shape
        
        print(f_spatio_temporal.shape,h,w)

        # Change ref_boxes to be in the shape of the feature map
        print('###################### Ref Boxes Stuff ##########################')
        f_ref = self.upsample_layer(f_ref)
        bs, c, h , w  = f_ref.shape
        ref_boxes = torch.stack(ref_boxes,dim=0)
        ref_boxes *= torch.Tensor((h,w,h,w))
        ref_boxes = ref_boxes.view(bs,-1,4)
        print(f_ref.shape,ref_boxes.shape,torch.unbind(ref_boxes, dim=0)[0].shape)
        # ROI Align the Reference Boxes of the Refernce Frame
        ref_boxes_features = torchvision.ops.roi_align(
            f_ref,
            list(torch.unbind(ref_boxes, dim=0)),
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 7, 7)

        print('ref boxes shape',ref_boxes_features.shape)

        

        # Initializinf Track Queries using Ref Boxes Features
        # What does the Transformer is suposed to do ?
        # Use Reference Boxes to initalize Spatial Query, Ref Boxes Features to Initalize Content Query

        # Upsampling




        print('Getting into Heads')





        # Send to Single Object Head and use that as teacher for student network ead


        # Fuse these layers with input features map for t
        #################################### Tracking Stuff #####################################

        

        #
        
        return ...

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4,weight_decay= 1e-4) 
        scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 10)   
        return {"optimizer":optimizer, "lr_scheduler":scheduler}

    def training_step(self,batch, batch_idx):
        print(batch.keys())
        print(batch['image'][0].shape, batch['pre_img'][0].shape)
        
        
        out = self.forward(batch)
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
#     def forward(self, track_instances: Instances, target_size) -> Instances:
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation)
#                           For visualization, this should be the image size after data augment, but before padding
#         """
#         out_logits = track_instances.pred_logits
#         out_bbox = track_instances.pred_boxes

#         prob = out_logits.sigmoid()
#         # prob = out_logits[...,:1].sigmoid()
#         scores, labels = prob.max(-1)

#         # convert to [x0, y0, x1, y1] format
#         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
#         # and from relative [0, 1] to absolute [0, height] coordinates
#         img_h, img_w = target_size
#         scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
#         boxes = boxes * scale_fct[None, :]

#         track_instances.boxes = boxes
#         track_instances.scores = scores
#         track_instances.labels = labels
#         track_instances.remove('pred_logits')
#         track_instances.remove('pred_boxes')
#         return track_instances