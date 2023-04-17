import torch
import torch.nn as nn

import sys

sys.path.append("../")  ## TODO : cleaner way to import module in parent directory?
from global_utils import IoU


class YOLOv3loss(nn.Module):
    def __init__(self, lambda_class = 1, lambda_obj = 10, lambda_noobj = 1, lambda_coord=10):
        super().__init__()
        
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_coord = lambda_coord
        
        self.MSEloss = nn.MSELoss(reduction="mean")
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.CEloss = nn.CrossEntropyLoss()

        self.sigmoid = nn.Sigmoid()

        
    def forward(self, predictions, target, anchors):
        
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.BCEloss(
            (predictions[..., 0:1][noobj], target[..., 0:1][noobj])
        )
        # Object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2) # p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim = -1)
        ious = IoU(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce( predictions[..., 0:1][obj], (ious * target[..., 0:1][obj]))

        
        # Box Coordinate Loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(
            1e-16 + target[..., 3:5] / anchors
        )
        box_loss = self.MSEloss(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # class loss
        class_loss = self.CEloss(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long())
        )

        return (
            self.lambda_coord * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )