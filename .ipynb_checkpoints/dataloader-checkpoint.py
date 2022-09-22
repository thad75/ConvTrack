import pytorch_lightning as pl
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.mot20 import build_mot2020
from util.misc import mot_collate_fn as collate_fn
from util.image import GaussianBlur

class MOT2020Module(pl.LightningDataModule):
    def __init__(self,folder,
                    cache_mode = False, 
                    input_h= 480 , 
                    input_w =960 ,
                    image_blur_aug = True,
                    flip = 0,
                    not_max_crop = True,
                    blur_aug =  GaussianBlur(kernel_size=11),
                    not_rand_crop = True,
                    shift=0,
                    scale = 0, 
                    rotate = 0,
                    aug_rot = 0,
                    same_aug_pre = True,
                    no_color_aug = True,
                    max_frame_dist= 4,
                    down_ratio = 4,
                    pre_hm = True,
                    num_classes = 1,
                    hm_disturb = 0.05,
                    lost_disturb = 0.4,
                    fp_disturb = 0.1,
                    batch_size=2):
        """
        No test annotation ?
        
        
        """
        # TOOD : Add Half Split Scheme
        super().__init__()
        self.batch_size = batch_size
        self.folder = folder
        self.input_h = input_h
        self.input_w = input_w
        self.output_h = input_h
        self.output_w = input_w
        self.flip = flip
        self.image_blur_aug = image_blur_aug
        self.blur_aug = blur_aug
        self.not_max_crop = not_max_crop 
        self.not_rand_crop = not_rand_crop
        self.shift= shift
        self.scale = scale
        self.rotate = rotate
        self.flip = flip    
        self.aug_rot = aug_rot
        self.same_aug_pre = same_aug_pre
        self.no_color_aug = no_color_aug
        self.max_frame_dist= max_frame_dist
        self.down_ratio = down_ratio
        self.pre_hm = pre_hm
        self.num_classes = num_classes
        self.hm_disturb = hm_disturb
        self.lost_disturb = lost_disturb
        self.fp_disturb = fp_disturb
        self.cache_mode = cache_mode

    def prepare(self):
        build_mot2020(folder='train')
        build_mot2020(folder='test') 
        
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.COCOtrain =  build_mot2020(self.folder, 
                                            split ='train'  , 
                                            cache_mode =self.cache_mode ,
                                            input_h=self.input_h  , 
                                            input_w =self.input_w  ,
                                            image_blur_aug =self.image_blur_aug ,
                                            flip =self.flip,
                                            not_max_crop =self.not_max_crop,
                                            blur_aug =self.blur_aug,
                                            not_rand_crop =self.not_rand_crop ,
                                            shift=self.shift,
                                            scale =self.scale, 
                                            rotate =self.rotate,
                                            aug_rot =self.aug_rot,
                                            same_aug_pre =self.same_aug_pre,
                                            no_color_aug =self.no_color_aug,
                                            max_frame_dist=self.max_frame_dist,
                                            down_ratio =self.down_ratio,
                                            pre_hm =self.pre_hm,
                                            num_classes =self.num_classes,
                                            hm_disturb =self.hm_disturb,
                                            lost_disturb =self.lost_disturb,
                                            fp_disturb =self.fp_disturb)
             
        if stage == "test" or stage is None:
            # TODO : Change after
            self.COCOtest = build_mot2020(self.folder, 
                                            split ='test'  , 
                                            cache_mode =self.cache_mode ,
                                            input_h=self.input_h  , 
                                            input_w =self.input_w  ,
                                            image_blur_aug =self.image_blur_aug ,
                                            flip =self.flip,
                                            not_max_crop =self.not_max_crop,
                                            blur_aug =self.blur_aug,
                                            not_rand_crop =self.not_rand_crop ,
                                            shift=self.shift,
                                            scale =self.scale, 
                                            rotate =self.rotate,
                                            aug_rot =self.aug_rot,
                                            same_aug_pre =self.same_aug_pre,
                                            no_color_aug =self.no_color_aug,
                                            max_frame_dist=self.max_frame_dist,
                                            down_ratio =self.down_ratio,
                                            pre_hm =self.pre_hm,
                                            num_classes =self.num_classes,
                                            hm_disturb =self.hm_disturb,
                                            lost_disturb =self.lost_disturb,
                                            fp_disturb =self.fp_disturb)
            
    def train_dataloader(self):
        return DataLoader(self.COCOtrain, batch_size = self.batch_size, drop_last = False,shuffle = True, collate_fn= collate_fn)
    # def val_dataloader(self):
    #     return DataLoader(self.COCOvalid, batch_size = self.batch_size, drop_last =  False, collate_fn= collate_fn)        
    def test_dataloader(self):
        return DataLoader(self.COCOtest, batch_size = self.batch_size, drop_last =  False, collate_fn= collate_fn)