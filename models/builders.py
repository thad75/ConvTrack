from .criterion import SetCriterion
import torch 
import torch.nn as nn 


def build_weights(loss_ce= 2,  loss_bbox=5,  loss_giou= 2, loss_mask = 1,loss_similarity = 0.001, loss_dice = 1):
        weight_dict = { 'loss_ce':loss_ce,      
                       'loss_bbox': loss_bbox,
                       'loss_similarity': loss_similarity,
                       'loss_giou': loss_giou,        
                       'loss_dice': loss_dice,        
                       'loss_mask': loss_mask,
                       'loss_ce_dn':loss_ce,      
                       'loss_bbox_dn': loss_bbox,
                       'loss_giou_dn': loss_giou,        
                       'loss_dice_dn': loss_dice,  
                       'loss_similarity_dn': loss_similarity,
                       'loss_mask_dn': loss_mask, }            
        return weight_dict
    
    
def build_criterion(matcher , weight_dict,  aux_loss = False, num_classes= 91,dec_layers = 6 , masks = False):    
   
    if aux_loss:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality','similarity']
    if masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             focal_alpha=0.25,
                             losses=losses)
    return criterion

    
def build(num_queries = 300, backbone = 'resnet50', num_classes = 91, masks = False, aux_loss = False, dec_layers = 6, dataset_file = 'coco'):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    #if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
    #    num_classes = 250
    
    # Creating Model
    backbone = build_backbone(backbone = backbone)
    transformer = build_transformer()    
    model = FastDETR(num_queries = 300,
                 backbone = backbone,
                 transformer = transformer,
                num_classes = 91)
    if masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    # Creating Matcher and Criterion (Focal Loss, Bbox Loss)
    matcher = build_matcher()    
    weight_dict = build_weights()
    
    if aux_loss:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality']
    if masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             focal_alpha=0.25,
                             losses=losses)
    
    # Creating Post Processor
    post_processors = {'bbox': PostProcess()}
    if masks:
        post_processors['segm'] = PostProcessSegm()
        if dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            post_processors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, post_processors
