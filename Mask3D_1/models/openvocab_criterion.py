# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from models.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

def calculate_iou(mask1):
    predictions = mask1[:,1]
    targets = mask1[:,0]
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Calculate intersection and union
    intersection = torch.sum(predictions * targets, dim=1)
    union = torch.sum(predictions + targets, dim=1) + 1e-5

    # Calculate IoU
    iou = intersection / union
    #sorted, indices = torch.sort(intersection,descending=True)[:5]
    #sorted1, indices1 = torch.sort(union,descending=True)[:5]
    #sorted2, indices2 = torch.sort(iou,descending=True)[:5]

    #if(iou != 0):
    #    print("OU")
    return iou


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    iou_mask
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    mult = (iou_mask * loss)
    return mult.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    iou_mask
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    inputs = F.softmax(inputs,dim=-1)
    loss = F.binary_cross_entropy(
        inputs, targets, reduction="none"
    )
    mult = (iou_mask.reshape(-1,1) * loss)
    mult = torch.clamp(mult, min=0, max=5)

    return mult.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

#tens1 = torch.tensor([[1,0,0,0,1000],[0,5,0,0,0]],dtype=torch.float32)
#tens2 = torch.tensor([[0,1,0,0,0],[0,1,0,0,0]],dtype=torch.float32)
#sigmoid_ce_loss(tens1,tens1,5,tens1)
def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class OpenVocabSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, num_masks, mask_type):
        # todo this will have to be modified to cos dist or something
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        
        losses = []
        
        for i in range(outputs["pred_logits"].shape[0]):
            out = outputs["pred_masks"][i]# @ outputs["pred_masks_1"][i]
            segment_features = targets[i]["segment_mask"].to(torch.float32) @ out
            #print(torch.min(out),torch.max(out))
            #print(torch.min(segment_features),torch.max(segment_features))
            counts = torch.sum(targets[i]["segment_mask"],dim=-1).reshape(-1,1)
            ######more than nr of segments in count -> more than one instance per segment
            feats_normalized = segment_features/counts
            #feats_normalized = torch.nn.functional.sigmoid(feats_normalized)
            target = torch.tensor(targets[i]["instance_feats"],device="cuda")
            #target = torch.nn.functional.sigmoid(target)
            #print(torch.min(feats_normalized),torch.max(feats_normalized))
            #print(torch.min(target),torch.max(target))
            
            sim = torch.nn.functional.cosine_similarity(target,feats_normalized) 
            cosine_sim = 1 - torch.mean(sim)
            losses.append(cosine_sim)
            
        losses = torch.stack(losses).mean()
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat(
        #     [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        # )
        # target_classes = torch.full(
        #     src_logits.shape[:2],
        #     self.num_classes,
        #     dtype=torch.int64,
        #     device=src_logits.device,
        # )
        # target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(
        #     src_logits.transpose(1, 2),
        #     target_classes,
        #     self.empty_weight,
        #     ignore_index=253,
        # )
        losses = {"loss_cos": losses}
        return losses

    def loss_masks(self, outputs, targets, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []
        foreground_threshold = 0.001
        for i in range(outputs["pred_logits"].shape[0]):
            map_ = outputs["pred_masks"][i] @ outputs["pred_logits"][i]
            target_mask = targets[i][mask_type]
            #print(torch.min(map_),torch.max(map_))
            #print(torch.min(target),torch.max(target))
            foreground_mask = map_.softmax(dim=-1) > foreground_threshold
            map_ = map_.T


            masks = [torch.cat((foreground_mask.T[i].reshape(1,-1), target_mask[j].reshape(1,-1)),dim=0) for j in range(len(target_mask)) for i in range(len(map_))]
            masks_stacked = torch.stack(masks)
            iou = calculate_iou(masks_stacked)
            #print([iou_ for iou_ in iou if iou_ != 0])
            

            sorted, indices = torch.sort(iou,descending=True)

            ind = [torch.nonzero((indices / 100).to(torch.long) == target_value) for target_value in range(target_mask.shape[0])]
            indices_ = torch.tensor([torch.nonzero((indices / 100).to(torch.long) == target_value)[0] for target_value in range(target_mask.shape[0]) if len(torch.nonzero((indices / 100).to(torch.long) == target_value)) != 0])
            #indices_ = indices.expand(target_mask.shape[0],500)


            sorted = sorted[indices_]
            indices = indices[indices_]
            iou_mask = (sorted > 0.60)

            src_index = indices % 100
            tgt_index = (indices / 100).to(torch.long)

            map_ = map_[src_index]


            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map_ = map_[:, point_idx]
            target_mask = target_mask[:, point_idx].float()
            target_mask = target_mask[tgt_index]

            loss_masks.append(sigmoid_ce_loss(map_, target_mask, num_masks,iou_mask))
            loss_dices.append(dice_loss(map_, target_mask, num_masks,iou_mask))
        # del target_mask
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)) + 1e-5,
            "loss_dice": torch.sum(torch.stack(loss_dices)) + 1e-5,
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, num_masks, mask_type):
        """
        helper that computes labels loss or masks loss given the loss name as str
        """
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks, mask_type)

    def forward(self, outputs, targets, mask_type):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        #indices = self.matcher(outputs_without_aux, targets, mask_type)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device="cuda",
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, num_masks, mask_type
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                #indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        num_masks,
                        mask_type,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
