# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import torch
import IoU


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolov8n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, gt=None):
        """Post-processes predictions and returns a list of Results objects."""
        # preds = ops.non_max_suppression(
        #     preds,
        #     self.args.conf,
        #     self.args.iou,
        #     agnostic=self.args.agnostic_nms,
        #     max_det=self.args.max_det,
        #     classes=self.args.classes,
        # )

        # if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        #     orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        #
        # results = []
        # for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
        #     pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        #     results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        # return results

        preds_tensor = preds[0]  # Shape: [1, 84, 6300]
        # denominator should be preds_tensor.shape[-1], but it somethime returns tensor instead of int
        if type(preds_tensor.shape[-1]) == torch.Tensor:
            den = preds_tensor.shape[-1].item()
        else:
            den = preds_tensor.shape[-1]
        row_index, col_index = divmod(torch.argmax(preds_tensor[0, 4:]).item(), den)
        best_bbox = preds_tensor[:, :4, col_index]  # this bbox has the format of x_c, y_c, w, h
        best_bbox = ops.xywh2xyxy(best_bbox.unsqueeze(0))[0]

        IoU_calculator = IoU.IoU_xyxy(tau=0.5)
        if gt is None:
            gt = torch.rand_like(best_bbox)
        z = IoU_calculator(torch.cat((best_bbox, gt), dim=1).float())

        return z