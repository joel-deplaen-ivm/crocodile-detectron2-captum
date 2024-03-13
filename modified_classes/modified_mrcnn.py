from typing import List, Tuple

import torch

from detectron2.layers import batched_nms
#from detectron2.modeling.roi_heads.box_head import BaseMaskRCNNHead
#from detectron2.modeling.roi_heads import ROIHeads
from detectron2.modeling.roi_heads.mask_head import BaseMaskRCNNHead

from detectron2.structures import Boxes, Instances


def mask_rcnn_inference_single_image(
    boxes,
    scores,
    masks,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    class_scores_only: bool
):
    """
    Single-image inference for Mask R-CNN. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `mask_rcnn_inference`, but with boxes, scores, masks, and image shapes
        per image.

    Returns:
        Same as `mask_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        masks = masks[valid_mask]

    scores = scores[:, :-1]
    class_scores = scores.clone()

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    masks = masks[filter_mask]
    class_scores = class_scores[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, masks, filter_inds, class_scores = (
        boxes[keep],
        scores[keep],
        masks[keep],
        filter_inds[keep],
        class_scores[keep],
    )

    if class_scores_only:
        boxes = boxes.detach()
        scores = scores.detach()
        masks = masks.detach()
        filter_inds = filter_inds.detach()

        # class_scores: tensor[N x K]
        return class_scores, filter_inds[:, 0]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_masks = masks
    result.class_scores = class_scores
    result.pred_classes = filter_inds[:, 1]

    return result, filter_inds[:, 0]


#class ModifiedMaskRCNNOutputLayers(BaseMaskRCNNHead):
#class ModifiedMaskRCNNOutputLayers(ROIHeads):
class ModifiedMaskRCNNOutputLayers(BaseMaskRCNNHead):
    def __init__(self, mask_rcnn_output_layers_instance):
        #AttributeError: cannot assign module before Module.__init__() call
        super().__init__()
        self.class_scores_only = False
        self.mask_rcnn_output_layers = mask_rcnn_output_layers_instance

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `mask_rcnn_inference_single_image`.
            list[Tensor]: same as `mask_rcnn_inference_single_image`.
        """
        boxes, scores, masks = predictions
        image_shapes = [x.image_size for x in proposals]
        return mask_rcnn_inference_single_image(
            boxes,
            scores,
            masks,
            image_shapes,
            self.mask_rcnn_output_layers.test_score_thresh,
            self.mask_rcnn_output_layers.test_nms_thresh,
            self.mask_rcnn_output_layers.test_topk_per_image,
            self.class_scores_only
        )
