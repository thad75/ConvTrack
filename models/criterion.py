import torch
import torch.nn as nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, reduce_dict)
import torch.nn.functional as F

from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss, sigmoid_focal_loss)

class SetCriterion(nn.Module):
    """ This class computes the loss for SAM-DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.lambd = 0.0051
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) \
                  * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def get_off_diagonal_elements(self,M):
        res = M.clone()
        res.diagonal(dim1=-1, dim2=-2).zero_()
        return res
    
    
    def normalize(self,pred_embedding):
        pred_embedding = pred_embedding - pred_embedding.min()
        return pred_embedding/pred_embedding.max()
    
    def loss_similarity(self,outputs, targets, indices, num_boxes, denoising = False):
        
#         idx = self._get_src_permutation_idx(indices)
        pred_embedding = outputs['pred_embedding']

#         print('Denoising Value ', denoising, pred_embedding.shape)
        if denoising == False: 
#             print('Into Similairty')
            bs, nquery,hidden = pred_embedding.shape
#             print('Pred Embedding', pred_embedding.shape)
#             pred_embedding = pred_embedding - pred_embedding.min()
#             pred_embedding = pred_embedding/pred_embedding.max()
            pred_embedding = self.normalize(pred_embedding)

#             similarity_matrix = self.normalize(similarity_matrix)

            pred_embedding_t = pred_embedding.clone().transpose(2,1)
            similarity_matrix = torch.bmm(pred_embedding,pred_embedding_t)
#             print(similarity_matrix.unique())
            del pred_embedding_t, pred_embedding
            id_matrix = torch.eye(similarity_matrix.shape[1], similarity_matrix.shape[2], device = similarity_matrix.device)
            id_matrix = torch.stack([id_matrix for i in range(bs)])
            diff = similarity_matrix - id_matrix
            off_diag = self.get_off_diagonal_elements(similarity_matrix)
            off_diag = off_diag.pow_(2).add_(-1).sum()
            on_diag = torch.diagonal(similarity_matrix).add_(-1).pow_(2).sum()
#             print(on_diag, off_diag)
#             print(on_diag, F.mse_loss(similarity_matrix, id_matrix))
            loss = on_diag + self.lambd*off_diag #+ F.mse_loss(similarity_matrix, id_matrix).sum()
#             print(loss)
            losses = {'loss_similarity':loss/get_world_size()}
        else : 
            losses = {'loss_similarity':torch.as_tensor(0.).to(pred_embedding.device)}
#         print()
#         print('Loss',losses)
#         print()
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,           
            'similarity': self.loss_similarity,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
#         print('Beggining of Forward', outputs['pred_embedding'].unique() )
#         print(torch.isnan(outputs['pred_embedding'].unique()).int().sum())
#         print('NON')
#         print()
#         print('Beginnig', outputs['pred_embedding'].shape)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        device=next(iter(outputs.values())).device
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}

       # Adding some Denoising
        dn_meta = outputs['dn_meta']
        #print(self.training)
#         print("####################### DN Loss ########################")

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
           # print("Bo njour")
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))
            
#             print('On Outputes Forward', outputs['pred_embedding'].unique() )

#             print(torch.isnan(outputs['pred_embedding'].unique()).int().sum())
#             print('NON')
# 
            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
#             print( output_known_lbs_bboxes['pred_logits'].shape)
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                if 'similarity' in loss : 
                    kwargs = {'denoising':True}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to(device)
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to(device)
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to(device)
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to(device)
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to(device)
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to(device)
            losses.update(l_dict)
#         print('Before Loss of Forward', outputs['pred_embedding'].unique())
#         print(torch.isnan(outputs['pred_embedding'].unique()).int().sum())
#         print('NON')        
#         print()
#         print("####################### Classic Loss ########################")

        for loss in self.losses:
#             print('Bunda',outputs.keys())
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
#         print('Out of Forward', outputs['pred_embedding'].unique())
#         print(torch.isnan(outputs['pred_embedding'].unique()).int().sum())
#         print('NON') 
#         print('###############################################')

#         print("####################### Aux Loss ########################")
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
#             print('I have auxiliary outputs')
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
#                 print(' Noprmal in AUX LOSS')

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                        
#                     print('Blabla', aux_outputs['pred_embedding'].unique())
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
#                 print(' DN Loss in AUX LOSS')
                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][i]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}
                        if 'similarity' in loss : 
                            kwargs = {'denoising':True}
                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to(device)
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to(device)
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to(device)
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to(device)
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to(device)
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to(device)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    
    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups