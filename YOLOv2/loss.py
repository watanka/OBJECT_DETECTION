import torch
import torch.nn as nn

import sys

sys.path.append("../")  ## TODO : cleaner way to import module in parent directory?
from global_utils import IoU


class YOLOLoss(nn.Module):
    def __init__(self, anchorbox, lambda_coord=5, lambda_noobj=0.5, num_grid=7):
        super().__init__()

        self.anchorbox = anchorbox
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_grid = num_grid
        self.numbox = len(self.anchorbox)
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
        # class loss is considered per grid cell. We loop over gt boxes of the gridcells.

        # for cx and cy
        gridratio = 1 / self.num_grid
        y = torch.arange(self.num_grid)
        x = torch.arange(self.num_grid)
        grid_y = grid_x.expand(batch_size, -1,-1)
        grid_x = grid_y.expand(batch_size, -1,-1)


        for gtboxidx in range(self.numbox):

            gt_label = target[..., gtboxidx, :]

            identity_obj = gt_label[..., 0:1]  # to remain the last dimension

            # ious = torch.zeros((batch_size, self.num_grid, self.num_grid, self.numbox))
            ious = torch.zeros_like(output[..., 0])

            ## Box 갯수만큼 IoU를 계산
            # select only one box with the highest IoU
            for boxidx in range(self.numbox):  
                pred_coords = output[..., boxidx, :]
                ious[..., boxidx] = IoU(pred_coords[..., 1:5], gt_coords[..., 1:5])

            # torch.max return max values of the selected axis and the indices of them. we are going to use these indices to select the highest IoU bboxes
            _, iou_mask = torch.max(ious, axis=-1, keepdim=True)  


            # (B, grid, grid, (5 + num_classes))
            selected_preds = torch.gather(output, -2, iou_mask.unsqueeze(-1).repeat(1,1,1,1, 5 + self.num_classes ))  

            cx = selected_preds[...,1].sigmoid() + grid_x
            cy = selected_preds[...,2].sigmoid() + grid_y
            pw = selected_preds[...,3].exp() *
            ph = selected_preds[...,4].exp() *
            # sigmoid(px) + cx



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

            ###################
            # COORDINATE LOSS #
            ###################
            coord_xy_loss, coord_wh_loss = self.calculate_boxloss(
                selected_pred_coords[..., 1:5], gt_coords[..., 1:5], identity_obj
            )
            coordinate_loss += coord_xy_loss + coord_wh_loss

        ##############
        # CLASS LOSS #
        ##############
        # if any gt object exists, the first grid cell pr(object) = 1. We don't want to take account for no object box' class.
        obj_exists = target[..., 0:1]
        class_loss = self.MSEloss(
            obj_exists * output[..., self.numbox * 5 :],
            obj_exists * target[..., self.numbox * 5 :],
        )

        loss = (
            coordinate_loss * self.lambda_coord
            + obj_loss
            + noobj_loss * self.lambda_noobj
            + class_loss
        )

        return loss

    def calculate_boxloss(self, pred, gt, identity_obj):
        """
        pred : (B x num_grid x num_grid x 5). selected with the highest IoU
        gt : (B x num_grid x num_grid x 5)
        identity_obj : whether object exists or not

        """

        ## COORDINATE LOSS
        # (x - xpred)**2 + (y - ypred) ** 2
        coord_xy_loss = self.MSEloss(
            identity_obj * torch.square(gt[..., :2]), pred[..., :2]
        )
        # (w**(1/2) - wpred**(1/2)) + (h**(1/2) - hpred(1/2))
        coord_wh_loss = self.MSEloss(
            identity_obj
            * torch.sign(pred[..., 2:4])
            * torch.sqrt(torch.abs(pred[..., 2:4])),
            torch.sqrt(gt[..., 2:4] + 1e-9),
        )

        return coord_xy_loss, coord_wh_loss
