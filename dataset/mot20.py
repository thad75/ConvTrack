from re import S
from util.image import GaussianBlur,affine_transform, draw_umich_gaussian, gaussian_radius,color_aug, get_affine_transform

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from os import listdir
from torch.utils.data import Dataset, DataLoader
#from heatmap_process import * 
from os import listdir
import os
import cv2 
from torchvision import transforms
#from mot20_process import Process_MOT20, Process_Heatmap_numpy
import random 
from PIL import Image
import numpy as np
import copy
import os 
import pycocotools.coco as coco
from tqdm import tqdm
from collections import defaultdict
import torch
import math
# TODO : Do MOT2020
# TODO : Same Transformations as in COCO

# https://github.com/dvlab-research/ECCV22-P3AFormer-Tracking-Objects-as-Pixel-wise-Distributions/blob/673d34698188e23e18e8ac920ec229ee79e67d71/datasets/p3aformer_dataset/generic_dataset_train.py#L310

class MOT2020(Dataset):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    _eig_vec = np.array(
        [
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938],
        ],
        dtype=np.float32,
    )
    cat_ids = {1: 1}
    num_joints = 17
    def __init__(self,folder, 
                    split = 'train', 
                    cache_mode = False, 
                    input_h= 480 , 
                    input_w =960 ,
                    image_blur_aug = True, 
                    flip = 0,
                    not_max_crop = True,
                    blur_aug =  GaussianBlur(kernel_size=11),
                    # TODO : Change this as class attributes
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
                    fp_disturb = 0.1):
        super().__init__()
        self.folder = folder
        self.split = split
        self.max_objs = 300
        if split == 'test':
            img_dir = os.path.join(self.folder, 'test')
            ann_path = "./MOT20/annotations/test.json"

        if split == 'train':
            img_dir = os.path.join(self.folder, r'train')            
            ann_path = "./MOT20/annotations/train.json"
        if split == 'train_half':
            img_dir = os.path.join(self.folder, r'train')
            ann_path = r"./MOT20/annotations/train_half.json"
        if split == 'val_half':
            img_dir = os.path.join(self.folder, r'train')
            ann_path = r"./MOT20/annotations/val_half.json"

        print(img_dir, ann_path)
        self.img_dir = img_dir
        self.coco = coco.COCO(ann_path)
# The above code is loading the COCO dataset and getting the image ids.
        self.images = self.coco.getImgIds()
        print('Number of Imges,',len(self.images))

        # print(self.images)
        self.tracking = True
        if self.tracking:
            if not ("videos" in self.coco.dataset):
                self.fake_video_data()
            print("Creating video index!")
            self.video_to_images = defaultdict(list)
            for image in self.coco.dataset["images"]:
                self.video_to_images[image["video_id"]].append(image)



        if cache_mode:
            self.cache = {}
            print("caching data into memory...")
            for tmp_im_id in tqdm(self.images):
                img, anns, img_info, img_path = self._load_image_anns(
                    tmp_im_id, self.coco, self.img_dir
                )
                assert tmp_im_id not in self.cache.keys()
                self.cache[tmp_im_id] = [img, anns, img_info, img_path]
        else:
            self.cache = {}

        self.input_h = input_h
        self.input_w = input_w
        self.output_h = input_h
        self.output_w = input_w
        self.flip = flip
        self.image_blur_aug = image_blur_aug
        self.blur_aug = blur_aug
        self.not_max_crop = not_max_crop 
        # TODO : Change this as class attributes
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
        print('MOT 2020 is initalized')
        # TODO : Change to class attribtues
        self.heads = ["hm", "reg", "wh", "center_offset", "tracking"]
        self.dense_reg = 1
        self.debug = 0
    def __getitem__(self,idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        img, anns, img_info, img_path, pad_img = self._load_data(idx)
        # If image is blurred :
        img_blurred = False

        if self.image_blur_aug and np.random.rand() < 0.1 and self.split == "train":
            img = self.blur_aug(img)
            img_blurred = True

        # Other Stuff
        height, width = img.shape[0], img.shape[1]
        # get image centers
        c = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32)
        # get image size or max h or max w
        s = (max(img.shape[0], img.shape[1]) * 1.0 if not self.not_max_crop
            else np.array([img.shape[1], img.shape[0]], np.float32))
        aug_s, rot, flipped = 1, 0, 0
        if self.split == "train":
            # drift image centers, change image size with scale, rotate image with rot.
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
            s = s * aug_s
            # random flip
            if np.random.random() < self.flip:
                flipped = 1
                img = img[:, ::-1, :].copy()
                anns = self._flip_anns(anns, width)
        trans_input = get_affine_transform(c, s, rot, [self.input_w, self.input_h])
        # the output heatmap size != input size, trans_output = transform for resizing gt to output size
        trans_output = get_affine_transform(c, s, rot, [self.output_w, self.output_h])
        inp, padding_mask = self._get_input(img, trans_input, padding_mask=pad_img)
        # plot
        ret = {"image": inp, "pad_mask": padding_mask.astype(np.bool)}
        gt_det = {"bboxes": [], "scores": [], "clses": [], "cts": []}


        # On pre_images or target frame
        pre_image,pre_anns,frame_dist,pre_img_id,pre_pad_image = self._load_pre_data(img_info["video_id"],
                                                                                    img_info["frame_id"],
                                                                                    img_info["sensor_id"] if "sensor_id" in img_info else 1)

        # All the augmentation on the pref image
        if self.image_blur_aug and img_blurred and self.split == "train":
                # print("blur image")
                pre_image = self.blur_aug(pre_image)

        if flipped:
            pre_image = pre_image[:, ::-1, :].copy()
            pre_anns = self._flip_anns(pre_anns, width)
        # if same_aug_pre and pre_img != curr_img, we use the same data aug for this pre image.
        if self.same_aug_pre and frame_dist != 0:
            trans_input_pre = trans_input.copy()
            trans_output_pre = trans_output.copy()
        else:
            c_pre, aug_s_pre, _ = self._get_aug_param(
                c.copy(), copy.deepcopy(s), width, height, disturb=True
            )
            s_pre = s * aug_s_pre
            trans_input_pre = get_affine_transform(
                c_pre, s_pre, rot, [self.input_w, self.input_h]
            )
            trans_output_pre = get_affine_transform(
                c_pre, s_pre, rot, [self.output_w, self.output_h]
            )

        # transform pre_image as standard input shape, todo warning pre_anns are not yet transformed
        pre_img, pre_padding_mask = self._get_input(
            pre_image, trans_input_pre, padding_mask=pre_pad_image
        )
        # pre_hm is of standard input shape, todo pre_cts is in the output image plane
        # Was added
        
        init_bbox= self._get_pre_bbox_for_roi(pre_anns, im_size = pre_image.shape)
        ret['init_bbox'] = np.array(init_bbox)
        
        pre_hm, pre_cts, pre_track_ids = self._get_pre_dets(
            pre_anns, trans_input_pre, trans_output_pre, im_size = pre_image.shape
        )
        ret["pre_img"] = pre_img
        ret["pre_pad_mask"] = pre_padding_mask.astype(np.bool)
        if self.pre_hm:
            ret["pre_hm"] = pre_hm

### init samples
        self._init_ret(ret, gt_det)
        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann["category_id"]])
            if cls_id > self.num_classes or cls_id <= -999:
                continue
            # get ground truth bbox in the output image plane,
            # bbox_amodal do not clip by ouput image size, bbox is clipped, todo !!!warning!!! the function performs cxcy2xyxy
            bbox, bbox_amodal = self._get_bbox_output(ann["bbox"], trans_output, height, width)
            if cls_id <= 0 or ("iscrowd" in ann and ann["iscrowd"] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                # print('mask ignore or crowd.')
                continue

            # todo warning track_ids are ids at t-1
            self._add_instance(ret,gt_det,k,cls_id,bbox,bbox_amodal,ann,trans_output,aug_s,pre_cts,pre_track_ids)

        # if self.debug > 0:
        gt_det = self._format_gt_det(gt_det)
        meta = {
            "c": c,
            "s": s,
            "gt_det": gt_det,
            "img_id": img_info["id"],
            "img_path": img_path,
            "flipped": flipped,
        }
        ret["meta"] = meta

        ret["c"] = c
        ret["s"] = np.asarray(s, dtype=np.float32)
        ret["image_id"] = self.images[idx]
        ret["output_size"] = np.asarray([self.output_h, self.output_w])
        ret["orig_size"] = np.asarray([height, width])

        return ret

    def __len__(self):
        """
        `__len__` is a special method that returns the length of the object
        :return: The length of the list.
        """
        return len(self.images)

    def _coco_box_to_bbox(self, box):
        bbox = np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )
        return   bbox
    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
            ],
            dtype=np.float32,
        )
        for t in range(4):
            rect[t] = affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_h - 1)
        return bbox, bbox_amodal
    
    def _get_border(self, border, size):
        """
        It returns the largest power of 2 that is less than or equal to the border size
        
        :param border: The border to be added around the image
        :param size: the size of the image to be generated
        :return: The border is being returned.
        """
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_aug_param(self, c, s, width, height, disturb=False):
        """
        It takes in the center of the bounding box, the size of the bounding box, the width and height of
        the image, and a boolean value that determines whether or not to disturb the bounding box. 
        
        It then returns the center of the bounding box, the size of the bounding box, and the rotation of
        the bounding box. 
        
        The function is called in the following function:
        
        :param c: the center of the bounding box
        :param s: scale
        :param width: the width of the image
        :param height: the height of the image
        :param disturb: whether to disturb the original bounding box, defaults to False (optional)
        :return: The center, scale, and rotation of the image.
        """
        if (not self.not_rand_crop) and not disturb:
            sf = self.sf
            cf = self.cf

            if type(s) == float or type(s) == np.float64 or type(s) == np.float32:
                s = [s, s]
            c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            sf = self.scale
            cf = self.shift
            # print(s)
            if type(s) == float or type(s) == np.float64 or type(s) == np.float32:
                s = [s, s]
            c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.aug_rot:
            print("random rotate is activated.")
            rf = self.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

    def _load_image_anns(self, img_id, coco, img_dir):
        """
        > This function loads the image, annotations, and image info for a given image id
        
        :param img_id: the image id in the coco dataset
        :param coco: the COCO API object
        :param img_dir: the directory where the images are stored
        """
        """
        
        """
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_dir, file_name)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        # bgr=> rgb
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, anns, img_info, img_path

    def _load_data(self, index):
        """
        > The function loads the image and annotations from the COCO dataset, and then pads the image if
        the aspect ratio is not the same as the input aspect ratio
        
        :param index: the index of the image in the dataset
        """

        coco = self.coco
        img_dir = self.img_dir
        img_id = self.images[index]
        if img_id in self.cache.keys():
            img, anns, img_info, img_path = self.cache[img_id]
        else:
            img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)
        # padding before affine warping to prevent cropping

        h, w, c = img.shape
        target_ratio = 1.0 * self.input_w / self.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(img.dtype)
            new_img[:, :w, :] = img
            if "width" in img_info.keys():
                img_info["width"] = new_w
        else:
            new_img = img

        return new_img, anns, img_info, img_path, np.ones_like(img)

    def _load_pre_data(self, video_id, frame_id, sensor_id=1):
        img_infos = self.video_to_images[video_id]
        # If training, random sample nearby frames as the "previous" frame
        # If testing, get the exact prevous frame
        if "train" in self.split:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if abs(img_info["frame_id"] - frame_id) < self.max_frame_dist
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
        else:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if (img_info["frame_id"] - frame_id) == -1
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
            if len(img_ids) == 0:
                img_ids = [
                    (img_info["id"], img_info["frame_id"])
                    for img_info in img_infos
                    if (img_info["frame_id"] - frame_id) == 0
                    and (
                        not ("sensor_id" in img_info)
                        or img_info["sensor_id"] == sensor_id
                    )
                ]
        rand_id = np.random.choice(len(img_ids))

        img_id, pre_frame_id = img_ids[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)
        # print(frame_dist)
        if img_id in self.cache.keys():
            img, anns, _, _ = self.cache[img_id]
        else:
            img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)

        # padding before affine warping to prevent cropping
        h, w, c = img.shape
        target_ratio = 1.0 * self.input_w / self.input_h
        if 1.0 * w / h < target_ratio:
            new_w = int(target_ratio * h)
            new_img = np.zeros((h, new_w, c)).astype(img.dtype)
            new_img[:, :w, :] = img

        else:
            new_img = img
        return new_img, anns, frame_dist, img_id, np.ones_like(img) 
    
    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.dense_reg
        ret["hm"] = np.zeros(
            (self.num_classes, self.output_h, self.output_w), np.float32
        )
        ret["ind"] = np.zeros((max_objs), dtype=np.int64)
        ret["cat"] = np.zeros((max_objs), dtype=np.int64)
        ret["mask"] = np.zeros((max_objs), dtype=np.float32)
        # xyh #
        ret["boxes"] = np.zeros((max_objs, 4), dtype=np.float32)
        # Was Added
        # ret['init_boxes'] = np.zeros((max_objs, 4), dtype=np.float32)
        ret["boxes_mask"] = np.zeros((max_objs), dtype=np.float32)

        ret["center_offset"] = np.zeros((max_objs, 2), dtype=np.float32)

        regression_head_dims = {
            "reg": 2,
            "wh": 2,
            "tracking": 2,
            "ltrb": 4,
            "ltrb_amodal": 4,
            "nuscenes_att": 8,
            "velocity": 3,
            "hps": self.num_joints * 2,
            "dep": 1,
            "dim": 3,
            "amodel_offset": 2,
            "center_offset": 2,
        }

        for head in regression_head_dims:
            if head in self.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                ret[head + "_mask"] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                gt_det[head] = []

        if "hm_hp" in self.heads:
            num_joints = self.num_joints
            ret["hm_hp"] = np.zeros(
                (num_joints, self.output_h, self.output_w), dtype=np.float32
            )
            ret["hm_hp_mask"] = np.zeros((max_objs * num_joints), dtype=np.float32)
            ret["hp_offset"] = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
            ret["hp_ind"] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret["hp_offset_mask"] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32
            )
            ret["joint"] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if "rot" in self.heads:
            ret["rotbin"] = np.zeros((max_objs, 2), dtype=np.int64)
            ret["rotres"] = np.zeros((max_objs, 2), dtype=np.float32)
            ret["rot_mask"] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({"rot": []})
            
    def _get_pre_bbox_for_roi(self, anns, im_size= None):
        # for ann in anns:
            h,w,c = im_size
            bbox = torch.stack([torch.Tensor(self._coco_box_to_bbox(ann["bbox"])) for ann in anns])
            bbox[:,[0,2]] = bbox[:,[0,2]]/w
            bbox[:,[1,3]] = bbox[:,[1,3]]/h
            return bbox
        
    def _get_pre_dets(self, anns, trans_input, trans_output, im_size=None):
        hm_h, hm_w = self.input_h, self.input_w
        down_ratio = self.down_ratio
        trans = trans_input
        reutrn_hm = self.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            # print(im_size)
            cls_id = int(self.cat_ids[ann["category_id"]])
            if (
                cls_id > self.num_classes
                or cls_id <= -99
                or ("iscrowd" in ann and ann["iscrowd"] > 0)
            ):
                continue
            bbox = self._coco_box_to_bbox(ann["bbox"])
            # print(bbox)
            # from original input image size to standard input size using draw_umich_gaussian
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            # draw gt heatmap with
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                )
                ct0 = ct.copy()
                conf = 1
                # add some noise to ground-truth pre info
                ct[0] = ct[0] + np.random.randn() * self.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.hm_disturb * h
                conf = 1 if np.random.random() > self.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                # conf == 0, lost hm, FN
                track_ids.append(ann["track_id"] if "track_id" in ann else -1)
                if reutrn_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                # false positives disturb
                if np.random.random() < self.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)
        return pre_hm, pre_cts, track_ids

    def _flip_anns(self, anns, width):
        """
        > This function flips the annotations of the image

        :param anns: the annotations for the image
        :param width: the width of the image
        :return: The annotations are being returned.
        """
        for k in range(len(anns)):
            bbox = anns[k]["bbox"]
            anns[k]["bbox"] = [width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if "hps" in self.heads and "keypoints" in anns[k]:
                keypoints = np.array(anns[k]["keypoints"], dtype=np.float32).reshape(
                    self.num_joints, 3
                )
                keypoints[:, 0] = width - keypoints[:, 0] - 1
                for e in self.flip_idx:
                    keypoints[e[0]], keypoints[e[1]] = (
                        keypoints[e[1]].copy(),
                        keypoints[e[0]].copy(),
                    )
                anns[k]["keypoints"] = keypoints.reshape(-1).tolist()

            if "rot" in self.heads and "alpha" in anns[k]:
                anns[k]["alpha"] = (
                    np.pi - anns[k]["alpha"]
                    if anns[k]["alpha"] > 0
                    else -np.pi - anns[k]["alpha"]
                )

            if "amodel_offset" in self.heads and "amodel_center" in anns[k]:
                anns[k]["amodel_center"][0] = width - anns[k]["amodel_center"][0] - 1
        return anns

    def _get_input(self, img, trans_input, padding_mask=None):
        """
        It takes an image, transforms it, and returns the transformed image and a mask
        
        :param img: the image to be transformed
        :param trans_input: the affine transformation matrix
        :param padding_mask: This is a mask that is used to mask out the padding part of the image
        :return: The input image and the mask.
        """
        img = img.copy()
        if padding_mask is None:
            padding_mask = np.ones_like(img)
        inp = cv2.warpAffine(
            img,
            trans_input,
            (self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR,
        )

        # to mask = 1 (padding part), not to mask = 0
        affine_padding_mask = cv2.warpAffine(
            padding_mask,
            trans_input,
            (self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR,
        )
        affine_padding_mask = affine_padding_mask[:, :, 0]
        affine_padding_mask[affine_padding_mask > 0] = 1

        inp = inp.astype(np.float32) / 255.0
        if (
            self.split == "train"
            and not self.no_color_aug
            and np.random.rand() < 0.2
        ):
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp, 1 - affine_padding_mask
       
    def fake_video_data(self):
        """
        It adds a "video_id" and "frame_id" to each image, and a "track_id" to each annotation
        :return: The return value is a list of dictionaries. Each dictionary contains the following keys:
        """
        self.coco.dataset["videos"] = []
        for i in range(len(self.coco.dataset["images"])):
            img_id = self.coco.dataset["images"][i]["id"]
            self.coco.dataset["images"][i]["video_id"] = img_id
            self.coco.dataset["images"][i]["frame_id"] = 1
            self.coco.dataset["videos"].append({"id": img_id})

        if not ("annotations" in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset["annotations"])):
            self.coco.dataset["annotations"][i]["track_id"] = i + 1
            
            
            
    def _format_gt_det(self, gt_det):
        if len(gt_det["scores"]) == 0:
            gt_det = {
                "bboxes": np.array([[0, 0, 1, 1]], dtype=np.float32),
                "scores": np.array([1], dtype=np.float32),
                "clses": np.array([0], dtype=np.float32),
                "cts": np.array([[0, 0]], dtype=np.float32),
                "pre_cts": np.array([[0, 0]], dtype=np.float32),
                "tracking": np.array([[0, 0]], dtype=np.float32),
                "bboxes_amodal": np.array([[0, 0]], dtype=np.float32),
                "hps": np.zeros((1, 17, 2), dtype=np.float32),
            }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det            
            
            
    def _add_instance(self,ret,gt_det,k,cls_id,bbox,bbox_amodal,ann,trans_output,aug_s,pre_cts=None,pre_track_ids=None,):

        # box is in the output image plane, add it to gt heatmap
        
        h, w = bbox_amodal[3] - bbox_amodal[1], bbox_amodal[2] - bbox_amodal[0]
        h_clip, w_clip = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h_clip <= 0 or w_clip <= 0:
            return
        # print(k)
        radius = gaussian_radius((math.ceil(h_clip), math.ceil(w_clip)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
        # int(ct)
        ct_int = ct.astype(np.int32)

        # 'cat': categories of shape [num_objects], recording the cat id.
        ret["cat"][k] = cls_id - 1
        # 'mask': mask of shape [num_objects], if mask == 1, to train, if mask == 0, not to train.
        ret["mask"][k] = 1
        if "wh" in ret:
            # 'wh' = box_amodal size,of shape [num_objects, 2]
            ret["wh"][k] = 1.0 * w, 1.0 * h
            ret["wh_mask"][k] = 1
        # 'ind' of shape [num_objects],
        # indicating the position of the object = y*W_output + x in a heatmap of shape [out_h, out_w] #todo warning CT_INT
        ret["ind"][k] = ct_int[1] * self.output_w + ct_int[0]
        # the .xxx part of the kpts
        ret["reg"][k] = ct - ct_int
        ret["reg_mask"][k] = 1

        # center_offset
        ret["center_offset"][k] = (
            0.5 * (bbox_amodal[0] + bbox_amodal[2]) - ct[0],
            0.5 * (bbox_amodal[1] + bbox_amodal[3]) - ct[1],
        )

        ret["center_offset_mask"][k] = 1

        # ad pts to ground-truth heatmap
        # print("ct_int", ct_int)

        draw_umich_gaussian(ret["hm"][cls_id - 1], ct_int, radius)

        gt_det["bboxes"].append(
            np.array(
                [ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2],
                dtype=np.float32,
            )
        )
        # cx, cy, w, h
        # clipped box
        # ret['boxes'][k] = np.asarray([ct[0], ct[1], w, h], dtype=np.float32)
        ret["boxes"][k] = np.asarray(
            [
                0.5 * (bbox_amodal[0] + bbox_amodal[2]),
                0.5 * (bbox_amodal[1] + bbox_amodal[3]),
                (bbox_amodal[2] - bbox_amodal[0]),
                (bbox_amodal[3] - bbox_amodal[1]),
            ],
            dtype=np.float32,
        )

        # cx, cy, w, h / output size
        ret["boxes"][k][0::2] /= self.output_w
        ret["boxes"][k][1::2] /= self.output_h
        ret["boxes_mask"][k] = 1
        gt_det["scores"].append(1)
        gt_det["clses"].append(cls_id - 1)
        gt_det["cts"].append(ct)

        if "tracking" in self.heads:
            # if 'tracking' we produce ground-truth offset heatmap
            # if curr track id exists in pre track ids
            if ann["track_id"] in pre_track_ids:
                # get pre center pos
                pre_ct = pre_cts[pre_track_ids.index(ann["track_id"])]
                ret["tracking_mask"][k] = 1
                # todo 'tracking' of shape [# current objects, 2], be careful pre_ct (float) - CT_INT (the int part)
                # predict(ct_int) + ret['tracking'][k] = pre_ct (bring you to pre center)
                ret["tracking"][k] = pre_ct - ct_int
                gt_det["tracking"].append(ret["tracking"][k])
            else:
                gt_det["tracking"].append(np.zeros(2, np.float32))


# path = r".\MOT20\train"
# dataset = MOT2020(folder= path)
# print(dataset[0])
def build_mot2020(folder, 
                    split = 'train', 
                    cache_mode = False,
                    input_h= 480 , 
                    input_w =960 ,
                    image_blur_aug = True,
                    flip = 0,
                    not_max_crop = True,
                    blur_aug =  GaussianBlur(kernel_size=11),
                    # TODO : Change this as class attributes
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
                    fp_disturb = 0.1):
    return  MOT2020(folder, 
                    split =split  , 
                    cache_mode =cache_mode ,
                    input_h=input_h  , 
                    input_w =input_w  ,
                    image_blur_aug = image_blur_aug ,
                    flip = flip,
                    not_max_crop = not_max_crop,
                    blur_aug =  blur_aug,
                    not_rand_crop =not_rand_crop ,
                    shift=shift,
                    scale = scale, 
                    rotate = rotate,
                    aug_rot = aug_rot,
                    same_aug_pre = same_aug_pre,
                    no_color_aug = no_color_aug,
                    max_frame_dist= max_frame_dist,
                    down_ratio = down_ratio,
                    pre_hm = pre_hm,
                    num_classes = num_classes,
                    hm_disturb = hm_disturb,
                    lost_disturb = lost_disturb,
                    fp_disturb =fp_disturb)