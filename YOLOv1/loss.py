import torch
import torch.nn as nn

import sys
sys.path.append('../') ## TODO : cleaner way to import module in parent directory?
from global_utils import IoU


class YOLOLoss(nn.Module) :
    def __init__(self, lambda_coord = 5, lambda_noobj = 0.5, num_grid = 7, numBox = 2) :
        super().__init__()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_grid = num_grid
        self.numBox = numBox
        self.MSEloss = nn.MSELoss(reduction = 'mean')

        

    def forward(self, output, target) :
        '''
        for num_classes = 13, and numBox = 2,
        each grid cell contains 
        [[x,y,w,h,pr(obj), x,y,w,h,pr(obj), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],]

        '''
        batch_size = output.size(0)
        
        obj_loss = 0
        noobj_loss = 0
        coordinate_loss = 0
        # class loss is considered per grid cell. We loop over gt boxes of the gridcells.

        for gtbox_idx in range(self.numBox) :

            gt_coords = target[..., gtbox_idx*5 : (gtbox_idx+1)*5]

            identity_obj = gt_coords[..., -1:] # to remain the last dimension

            total_pred_coords = output[..., :self.numBox*5] # batch_size, grid, grid, (x,y,w,h,pr(obj) )

            # ious = torch.zeros((batch_size, self.num_grid, self.num_grid, self.numBox))
            ious = torch.zeros_like(total_pred_coords[..., :self.numBox])

            ## Box 갯수만큼 IoU를 계산
            for box_idx in range(self.numBox) : # select only one box with the highest IoU
                pred_coords = total_pred_coords[..., box_idx*5 : (box_idx + 1)*5]
                ious[..., box_idx:box_idx+1] = IoU(pred_coords, gt_coords)


            _, iou_mask = torch.max(ious, axis = -1, keepdim = True) # torch.max return max values of the selected axis and the indices of them. we are going to use these indices to select the highest IoU bboxes
            
            # iou mask turn into index slices
            iou_mask = torch.cat([iou_mask + i for i in range(5)], dim = -1)

            # print(total_pred_coords[iou_mask * 5 : (iou_mask+1) * 5].shape)


            


            selected_pred_coords = torch.gather(total_pred_coords, -1, iou_mask) # (B, grid, grid, 1)

            # object loss calculated for the box with the highest IoU
            # obj_loss = self.calculate_object_loss(selected_pred_coords[..., -1], gt_coords[], identity_obj)
            # no_obj loss calculated for entire boxes compare with the gt
            # no_obj_loss = self.calculate_no_obj_loss(total_pred_coords, gt_coords, identity_obj )


            ###############
            # OBJECT LOSS #
            ###############


            obj_loss += self.MSEloss(identity_obj * selected_pred_coords[..., -1:], identity_obj)

            ##################
            # NO_OBJECT LOSS #
            ##################
            # penalize if no object gt grid predicts bbox
            noobj_loss += self.MSEloss((1 - identity_obj) * selected_pred_coords[..., -1:], identity_obj)


            ###################
            # COORDINATE LOSS #
            ###################
            coord_xy_loss, coord_wh_loss = self.calculate_boxloss(selected_pred_coords, gt_coords, identity_obj)
            coordinate_loss += coord_xy_loss + coord_wh_loss

        ##############
        # CLASS LOSS #
        ##############
        # if any gt object exists, the first grid cell pr(object) = 1. We don't want to take account for no object box' class.
        obj_exists = target[..., 5:6]
        class_loss = self.MSEloss(obj_exists * output[..., self.numBox*5 : ], obj_exists * target[..., self.numBox*5 : ])


        loss = (coordinate_loss * self.lambda_coord \
                + obj_loss  \
                + noobj_loss * self.lambda_noobj \
                + class_loss)

        return loss



    def calculate_boxloss(self, pred, gt, identity_obj) :
        '''
        pred : (B x num_grid x num_grid x 5). selected with the highest IoU
        gt : (B x num_grid x num_grid x 5)
        identity_obj : whether object exists or not

        '''

        ## COORDINATE LOSS
        # (x - xpred)**2 + (y - ypred) ** 2
        coord_xy_loss = self.MSEloss(identity_obj * torch.square(gt[..., :2]), pred[..., :2]) 
        # (w**(1/2) - wpred**(1/2)) + (h**(1/2) - hpred(1/2))
        coord_wh_loss = self.MSEloss(identity_obj * torch.sign(pred[..., 2:4]) * torch.sqrt(torch.abs(pred[..., 2:4])), torch.sqrt(gt[..., 2:4] + 1e-9))

        return coord_xy_loss, coord_wh_loss
        
