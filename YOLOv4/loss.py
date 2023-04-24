import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss

import sys


sys.path.append("../")  ## TODO : cleaner way to import module in parent directory?
from global_utils import IoU, convert_boxformat

class FocalLoss(nn.Module):
    # https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/loss.py
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class IoULoss(nn.Module) :
    def __init__(self, method = 'IoU') :
        super().__init__()
        self.method = method

    def forward(self, inp, target) :
        '''
        input : (B, # of bboxes, 6)
        '''
        inp_w = inp[..., 2:3]
        inp_h = inp[..., 3:4]
        target_w = target[..., 2:3]
        target_h = target[..., 3:4]

        inp_area = inp_w * inp_h
        target_area = target_w * target_h

        inp_xmin = inp[..., 0:1] - inp_w / 2
        inp_ymin = inp[..., 1:2] - inp_h / 2
        inp_xmax = inp[..., 0:1] + inp_w / 2
        inp_ymax = inp[..., 1:2] + inp_w / 2

        target_xmin = target[..., 0:1] - target_w / 2
        target_ymin = target[..., 1:2] - target_h / 2
        target_xmax = target[..., 0:1] + target_w / 2
        target_ymax = target[..., 1:2] + target_w / 2
        
        inp_topleft = torch.cat([inp_xmin, inp_ymin], axis = -1)
        target_topleft = torch.cat([target_xmin, target_ymin], axis = -1)

        inp_bottomright = torch.cat([inp_xmax, inp_ymax], axis = -1)
        target_bottomright = torch.cat([target_xmax, target_ymax], axis = -1)

        intersection_top_left = torch.max(inp_topleft, target_topleft)
        intersection_bottom_right = torch.min(inp_bottomright, target_bottomright)

        area_inter = torch.prod(
            torch.clip(intersection_bottom_right - intersection_top_left, min = 0 , max = None), -1).unsqueeze(-1)

        iou = area_inter / (inp_area + target_area - area_inter + 1e-9)

        if self.method == 'IoU' : 
            return 1 - iou

        elif self.method == 'GIoU' or self.method == 'DIoU' or self.method == 'CIoU' :
            # GIoU : IoU - |C \ (A U B)| over C. C는 bbox와 GT를 모두 포함하는 최소 크기의 박스.
            C_top_left = torch.min(inp_topleft, target_topleft)
            C_bottom_right = torch.max(inp_bottomright, target_bottomright)
            C_area = torch.prod(C_bottom_right - C_top_left, -1).unsqueeze(-1)

            if self.method == 'GIoU' :
                return 1 - (iou - (C_area - (inp_area + target_area - area_inter)) / C_area)

            # DIoU : 중심좌표 반영. 1 - IoU + euclidean(pred_center, gt_center) / (diagonal length of C)**2 . C는 bbox와 GT를 모두 포함하는 최소 크기의 박스.
            euclidean = torch.sqrt(torch.sum((inp[..., 0:2] - target[..., 0:2]) ** 2, dim = -1)).unsqueeze(-1)
            diagonal_length_C = torch.sum((C_bottom_right - C_top_left) ** 2, dim = -1).unsqueeze(-1)

            if self.method == 'DIoU' :
                return 1 - iou + (euclidean / diagonal_length_C)
            # CIoU : overlap area, central point distance, aspect ratio 고려. 
            # 1 - IoU + 1 - IoU + euclidean(pred_center, gt_center) / (diagonal length of C)**2 + aspect_ratio_resemblance * alpha
            # aspect_ratio_resemblance = 4 / pi**2 (arctan(w_gt/h_gt) - arctan(w_pred/h_pred)) ** 2. 
            # (4/pi**2) * (arctan(w/h)) range from -0.5 to 0.5
            # alpha = positive trade-off parameter. aspect_ratio_resemblance / (1-IoU) + aspect_ratio_resemblance. IoU가 클수록 aspect_ratio_resemblance의 영향력을 키운다.
            aspect_ratio_resemblance = (4 / torch.pi ** 2) * (torch.atan(target_w / target_h) - torch.atan(inp_w / inp_h)) ** 2
            alpha = aspect_ratio_resemblance / ( (1 - iou) + aspect_ratio_resemblance)

            if self.method == 'CIoU' :
                return 1 - iou + (euclidean / diagonal_length_C) + alpha * aspect_ratio_resemblance
        else :
            raise ValueError("loss method should be one of {IoU, GIoU, DIoU, CIoU}")

            
            
            

class YOLOv4loss(nn.Module):
    def __init__(self, lambda_class = 1, lambda_obj = 10, lambda_noobj = 1, lambda_coord=10):
        super().__init__()
        
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_coord = lambda_coord
        
        self.MSEloss = nn.MSELoss(reduction = 'mean')
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.CEloss = nn.CrossEntropyLoss(reduction = 'mean')
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, predictions, target, anchors):
        
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.BCEloss(
            predictions[..., 0:1][noobj], target[..., 0:1][noobj]
        )
        # Object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2) # p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim = -1)
        ious = IoU(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.BCEloss( predictions[..., 0:1][obj], (ious * target[..., 0:1][obj]))

        # CIoU loss
        # convert boxformat cxcy to xyxy before computing loss
        preds_xyxy = convert_boxformat(box_preds[obj], format = 'xyxy')
        target_xyxy = convert_boxformat(target[..., 1:5][obj], format = 'xyxy')
        ciou_loss = complete_box_iou_loss(preds_xyxy, target_xyxy, reduction = 'mean')

        
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
            + ciou_loss
        )