import torch
import torch.nn as nn

import sys

sys.path.append("../")  ## TODO : cleaner way to import module in parent directory?
from global_utils import IoU


class YOLOv2loss(nn.Module):
    def __init__(self, anchorbox, num_classes, lambda_coord=5, lambda_noobj=0.5, num_grid=7):
        super().__init__()

        self.anchorbox = torch.tensor(anchorbox)
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_grid = num_grid
        self.numbox = len(self.anchorbox)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss(reduction="mean")

    def forward(self, output, target):
        """
        for num_classes = 13, and numbox = 2,
        each grid cell contains
        [[x,y,w,h,pr(obj), x,y,w,h,pr(obj), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],]

        """
        batch_size = output.size(0)

        obj_loss = 0
        noobj_loss = 0
        coordinate_loss = 0
        class_loss = 0

        # for cx and cy
        gridratio = 1 / self.num_grid
        y = torch.arange(self.num_grid)
        x = torch.arange(self.num_grid)
        grid_y, grid_x = torch.meshgrid(x,y, indexing = 'ij')
        grid_y = grid_x.expand(batch_size, -1,-1) * gridratio
        grid_x = grid_y.expand(batch_size, -1,-1) * gridratio

        # ph, pw
        anchorbox_h = self.anchorbox[:,0]
        anchorbox_w = self.anchorbox[:,1]



        for gtboxidx in range(self.numbox):

            gt_label = target[..., gtboxidx, :]

            identity_obj = gt_label[..., 0:1]  

            # ious = torch.zeros((batch_size, self.num_grid, self.num_grid, self.numbox))
            ious = torch.zeros_like(output[..., 0])

            ## Box 갯수만큼 IoU를 계산
            # select only one box with the highest IoU
            for boxidx in range(self.numbox):  
                pred_coords = output[..., boxidx, :]
                print(pred_coords[..., 1:5].shape, gt_label[..., 1:5].shape)
                ious[..., boxidx: boxidx+1] = IoU(pred_coords[..., 1:5], gt_label[..., 1:5])

            # torch.max return max values of the selected axis and the indices of them. we are going to use these indices to select the highest IoU bboxes
            _, iou_mask = torch.max(ious, axis=-1, keepdim=True)  


            # (B, grid, grid, (5 + num_classes))
            selected_preds = torch.gather(output, -2, iou_mask.unsqueeze(-1).repeat(1,1,1,1, 5 + self.num_classes ))  


            ###################
            # COORDINATE LOSS #
            ###################
            pred_cx = selected_preds[...,1:2].sigmoid() + grid_x
            pred_cy = selected_preds[...,2:3].sigmoid() + grid_y
            pred_pw = selected_preds[...,3:4].exp() * torch.take(anchorbox_w, iou_mask)
            pred_ph = selected_preds[...,4:5].exp() * torch.take(anchorbox_h, iou_mask)

            gt_cx = gt_label[...,1:2]
            gt_cy = gt_label[...,2:3]
            gt_pw = gt_label[...,3:4]
            gt_ph = gt_label[...,4:5]

            coordinate_loss += identity_obj * ((gt_cx - pred_cx)**2 + (gt_cy - pred_cy)**2)
            coordinate_loss += identity_obj * ((torch.sqrt(gt_cx - pred_cx))**2 + (torch.sqrt(gt_cy - pred_cy))**2)

            ###############
            # OBJECT LOSS #
            ###############
            obj_loss += self.MSEloss(
                identity_obj * selected_pred[..., 0:1], identity_obj
            )

            ##################
            # NO_OBJECT LOSS #
            ##################
            # penalize if no object gt grid predicts bbox
            noobj_loss += self.MSEloss(
                (1 - identity_obj) * selected_pred[..., 0:1], identity_obj
            )

            

            ##############
            # CLASS LOSS #
            ##############
            class_loss += self.MSEloss(
                identity_obj * selected_pred[..., self.numbox * 5 :],
                identity_obj * gt_label[..., self.numbox * 5 :],
            )

        loss = (
            coordinate_loss * self.lambda_coord
            + obj_loss
            + noobj_loss * self.lambda_noobj
            + class_loss
        )

        return loss

