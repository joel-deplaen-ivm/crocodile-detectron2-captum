from typing import List, Tuple

import torch

from detectron2.layers import batched_nms
from detectron2.modeling.roi_heads.mask_head import BaseMaskRCNNHead
from detectron2.structures import Boxes, Instances


def mask_rcnn_bbox_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    masks: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `mask_rcnn_bbox_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i.
        masks (list[Tensor]): A list of Tensors of predicted masks for each image.
            Element i has shape (Ri, M, H, W), where Ri is the number of predicted objects
            for image i, M is the number of classes, and H, W are height and width of the masks.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
    """
    result_per_image = [
        mask_rcnn_bbox_inference_single_image(
            boxes_per_image, scores_per_image, masks_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, masks_per_image, image_shape in zip(
            scores, boxes, masks, image_shapes
        )
    ]

    return [x for x in result_per_image]


def mask_rcnn_bbox_inference_single_image(
    boxes,
    scores,
    masks,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference for bounding box prediction. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `mask_rcnn_bbox_inference`, but with boxes, scores, masks, and image shapes
        per image.

    Returns:
        Instances: Instances object containing predicted bounding boxes.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        masks = masks[valid_mask]

    num_bbox_reg_classes = boxes.shape[1] // 4
    scores = scores[:, :-1]

    # Filter results based on detection scores
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()
    boxes = Boxes(boxes[filter_inds[:, 0]])
    scores = scores[filter_mask]

    # Apply NMS
    keep = batched_nms(boxes.tensor, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]

    instances = Instances(image_shape)
    instances.pred_boxes = boxes[keep]
    instances.scores = scores[keep]

    return instances


class ModifiedMaskRCNNOutputLayers(BaseMaskRCNNHead):
    #AttributeError: cannot assign module before Module.__init__() call
    def __init__(self, mask_rcnn_output_layers_instance):
        self.class_scores_only = False
        super().__init__(input_shape = mask_rcnn_output_layers_instance.cls_score.in_features,
                         box2box_transform = mask_rcnn_output_layers_instance.box2box_transform,
                         num_classes = mask_rcnn_output_layers_instance.num_classes,
                         test_score_thresh = mask_rcnn_output_layers_instance.test_score_thresh,
                         test_nms_thresh = mask_rcnn_output_layers_instance.test_nms_thresh,
                         test_topk_per_image = mask_rcnn_output_layers_instance.test_topk_per_image,
                         cls_agnostic_bbox_reg = mask_rcnn_output_layers_instance.num_classes == 1,
                         smooth_l1_beta = mask_rcnn_output_layers_instance.smooth_l1_beta,
                         box_reg_loss_type = mask_rcnn_output_layers_instance.box_reg_loss_type,
                         loss_weight = mask_rcnn_output_layers_instance.loss_weight
                        )
    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `mask_rcnn_bbox_inference`.
        """
        boxes, scores, masks = predictions
        image_shapes = [x.image_size for x in proposals]
        return mask_rcnn_bbox_inference(
            boxes,
            scores,
            masks,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )
