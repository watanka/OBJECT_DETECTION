"""
Implementation of Yolo (v2) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl

import torchvision
import torchmetrics
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from backbone import BaseBlock, DarkNetBottleneck, CSPResnet, SPP, DarkNet53
from loss import YOLOv4loss
from utils import  decode_labelgrid
import sys

sys.path.append("../")
from global_utils import drawboxes, nms

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",

]



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act :
            return self.leakyrelu(self.bn(self.conv(x)))
        else :
            return self.conv(x)

class ResidualBlock(nn.Module) :
    def __init__(self, channels, use_residual = True, num_repeats = 1)  :
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats) :
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size = 1),
                    CNNBlock(channels//2, channels, kernel_size = 3, padding = 1)
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x) :
        for layer in self.layers :
            x = layer(x) + x if self.use_residual else layer(x)
        return x


class ScalePrediction(nn.Module) :
    def __init__(self, in_channels, num_classes) :
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size = 3, padding = 1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act = False, kernel_size = 1)
        )
        self.num_classes = num_classes

    def forward(self, x) :
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2) # batch_size, 3, numgrid, numgrid, num_classes + 5
        )



class Yolov4(pl.LightningModule) :
    def __init__(self, in_channels_list, multiscales, anchorbox,  num_classes) :
        super().__init__()
        '''
        backbone : list of 3
        top-down :
        bottom-up :
        '''
        num_anchors = len(anchorbox)
        self.multiscales = multiscales
        self.final_channels = num_anchors * (5 + num_classes)

        scaled_anchorbox = torch.tensor(anchorbox) * (torch.tensor(multiscales)[..., None, None].repeat(1, 3, 2))
        self.register_buffer('anchorbox', scaled_anchorbox)
        self.numbox = len(self.anchorbox) // len(self.multiscales)
        self.num_classes = num_classes


        self.yolo_loss = YOLOv4loss(
            lambda_class = 1,
            lambda_obj = 10,
            lambda_noobj = 1,
            lambda_coord = 10,
        )

        self.mAP = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
            iou_thresholds=None,
        )  # iou_thresholds = None is same as [0.5, 0.05, 0.95]
        # self.mAP = MeanAveragePrecisionMetrics(gts, preds, iou_threshold_range, confidence_threshold)

    



        # backbone : [(cbl_3, SPP, clb_3), cbl, cbl] 
        # backbone output : [b1, b2, b3]
        self.backbone = DarkNet53(act_fn = 'mish', block_fn = DarkNetBottleneck, expansion = 2, csp_fn = CSPResnet,
                                    in_channels_list = [3, 32,  32, 64, 128,  256, 512], 
                                    num_blocks_list = [1,2,8,8,4])
        
        # [256, 512, 1024]
        x1_channel, x2_channel, x3_channel = in_channels_list


        self.backbone_func1 = BaseBlock(in_channels = x1_channel, out_channels = x1_channel, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# 256x76x76
        self.backbone_func2 = BaseBlock(in_channels = x2_channel, out_channels = x2_channel, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# 512x38x38
        # CBL_3, SPP, CBL_3
        self.backbone_func3 =  nn.Sequential(*[BaseBlock(x3_channel, x3_channel, act_fn = 'leakyrelu') for _ in range(5)],
                                                SPP(), # SPP expand channels 4 times larger.
                                                BaseBlock(x3_channel * 4, x3_channel)
                                            )# 1024x19x19

        self.path_func1 = nn.Sequential(*[BaseBlock(x3_channel, x3_channel, act_fn = 'leakyrelu') for _ in range(5)],
                                          nn.Upsample(scale_factor= 2, mode = 'bilinear'),
                                            BaseBlock(x3_channel, x3_channel // 2)
                                            )# CBL_5, UP, CBL
                        
        # CBL_5, UP, CBL
        self.path_func2 = nn.Sequential(*[BaseBlock(x2_channel, x2_channel, act_fn = 'leakyrelu') for _ in range(5)],
                                          nn.Upsample(scale_factor= 2, mode = 'bilinear'),
                                            BaseBlock(x2_channel, x2_channel // 2)
                                            )# CBL_5, UP, CBL
        
        self.head_func1 = BaseBlock(in_channels = x1_channel, out_channels = x1_channel * 2, kernel_size = 1, stride = 2, padding = 0, act_fn = 'leakyrelu' )# CBL
        self.head_func2_1 = nn.Sequential(*[BaseBlock(x2_channel, x2_channel, act_fn = 'leakyrelu') for _ in range(5)])# CBL_5
        self.head_func2_2 = BaseBlock(in_channels = x2_channel, out_channels = x2_channel * 2, kernel_size = 1, stride = 2, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL
        self.head_func3 = nn.Sequential(*[BaseBlock(x3_channel, x3_channel, act_fn = 'leakyrelu') for _ in range(5)])# CBL_5

        self.result_func1 = BaseBlock(in_channels = x1_channel, out_channels = self.final_channels, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL
        self.result_func2 = BaseBlock(in_channels = x2_channel, out_channels = self.final_channels, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL
        self.result_func3 = BaseBlock(in_channels = x3_channel, out_channels = self.final_channels, kernel_size = 1, stride = 1, padding = 0, act_fn = 'leakyrelu' )# CBL# CBL


    # top-down : [(cbl_5, up, cbl), (cbl_5, up, cbl)]
    def top_down(self, b1, b2, b3, func1, func2) :
        '''
        b represents backbone output.
        b1 has the largest feature size and b3 has the smallest.
        func{num} corresponds to b{num}. 
        '''
        
        p3 = b3
        p2 = func1(p3) + b2
        p1 = func2(p2) + b1

        return (p1, p2, p3)

    def bottom_up(self, x1, x2, x3, func1, func2_1, func2_2, func3) :
        '''
        x1 has the largest feature size and x3 has the smallest.
        func{num} corresponds to x{num}. 
        func2 is splitted to func2_1 and func2_2 as the diagram
        '''
        r1 = x1
        r2 = func2_1((func1(x1) + x2))
        r3 = func3(func2_2(r2) + x3)

        return (r1, r2, r3)


    def forward(self, x) :
        b1, b2, b3 = self.backbone(x)
        b1 = self.backbone_func1(b1)
        b2 = self.backbone_func2(b2)
        b3 = self.backbone_func3(b3)

        # top down
        p1, p2, p3 = self.top_down(b1, b2, b3, self.path_func1, self.path_func2)
        
        # bottom up
        r1, r2, r3 = self.bottom_up(p1, p2, p3, self.head_func1, self.head_func2_1, self.head_func2_2, self.head_func3)
        
        # return shape (B, num_anchorbox, scale, scale, 5 + num_classes)
        output1 = self.result_func1(r1)
        
        output1 = self.result_func1(r1).reshape(r1.shape[0], 3, self.final_channels // 3, self.multiscales[0], self.multiscales[0]).permute(0,1,3,4,2)
        output2 = self.result_func2(r2).reshape(r2.shape[0], 3, self.final_channels // 3, self.multiscales[1], self.multiscales[1]).permute(0,1,3,4,2)
        output3 = self.result_func3(r3).reshape(r3.shape[0], 3, self.final_channels // 3, self.multiscales[2], self.multiscales[2]).permute(0,1,3,4,2)


        return (output1, output2, output3)

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay = 5e-4)
        scheduler = CosineAnnealingLR(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img_batch, label_grid = batch
        pred_multiscales = self.forward(img_batch)

        losses = []

        for scale_idx in range(len(self.multiscales)) :  
            loss = self.yolo_loss(predictions = pred_multiscales[scale_idx], target = label_grid[scale_idx], anchors = self.anchorbox[scale_idx])
            losses.append(loss)      
        

        mean_loss =  sum(losses) / len(losses)
        self.log("train_loss", mean_loss)    
        
        if batch_idx % 100 == 0 :
            with torch.no_grad() :
                bboxes_list = self.convert_bboxes(pred_multiscales)
                bboxes_after_nms = [nms(bboxes, confidence_threshold = 0.6, iou_threshold = 0.8) for bboxes in bboxes_list]

                grid = self.make_visgrid(bboxes_after_nms, img_batch)
                self.logger.experiment.add_image("bbox visualization", grid, self.global_step)

        return mean_loss

    def convert_bboxes(self, prediction_batch, is_preds = True) : 
        '''
        prediction_batch : [scale0, scale1, scale2]
            scale{num} shape : (batch_size, self.numbox, S, S, 5 + self.num_classes). 
            S differs by scale. [19, 38, 76]
        '''

        BATCH_SIZE = prediction_batch[0].shape[0]

        bboxes_list = [[] for batch_idx in range(BATCH_SIZE)]
        with torch.no_grad() :
            # convert predictions into bboxes
            # initiate bboxes_list to sort bbox by batch_idx
            for scale_idx in range(len(self.multiscales)) :
                bboxes = decode_labelgrid(prediction_batch[scale_idx],
                                            anchors = self.anchorbox[scale_idx],
                                            S = self.multiscales[scale_idx],
                                            is_preds = is_preds
                                            )
                for batch_idx in range(BATCH_SIZE) :
                    bboxes_list[batch_idx] += bboxes[batch_idx]

        return [np.array(bboxes) for bboxes in bboxes_list]

    def make_visgrid(self, bboxes_list, image_batch) :
        '''
        bboxes_list : [[obj, x, y, w, h, label] x ... ] x batch_size
        image batch : (batch_size, 3, imgsize, imgsize)
        '''
        with torch.no_grad() :
            # filter by nms and draw bbox
            bbox_visualization = []
            for img, bboxes in zip(image_batch, bboxes_list) :
                bbox_img = drawboxes(img, bboxes)
                bbox_visualization.append(torch.tensor(bbox_img))
            # hand over to to tensorboard
            grid_result = torch.stack(bbox_visualization).permute(0,3,1,2)
            grid = torchvision.utils.make_grid(grid_result)

        return grid
            


    def get_predgt(self, pred_bboxes, gt_bboxes) :
        '''
        bboxes to torchmetrics input format
        bboxes = [[conf_score, cx, cy, w, h, cls],...]
        '''
        if len(pred_bboxes) > 0 :
            pred_bboxes = torch.tensor(pred_bboxes)
            pred_conf, pred_coord, pred_cls = pred_bboxes[..., 0], pred_bboxes[..., 1:5], pred_bboxes[..., -1]
        else :
            pred_conf, pred_coord, pred_cls = torch.tensor([]), torch.tensor([]), torch.tensor([])
        if len(gt_bboxes) > 0 :
            gt_bboxes = torch.tensor(gt_bboxes)
            gt_obj, gt_coord, gt_cls = gt_bboxes[..., 0], gt_bboxes[..., 1:5], gt_bboxes[..., -1]
        else :
            gt_obj, gt_coord, gt_cls = torch.tensor([]), torch.tensor([]), torch.tensor([])

        preds = [
            dict(
                boxes = pred_coord,
                scores = pred_conf,
                labels = pred_cls,
            )
        ]

        target = [
            dict(
                boxes = gt_coord,
                labels = gt_cls,
            )
        ]

        return preds, target


    def validation_step(self, batch, batch_idx):
        ### 100번마다 한 번씩만 validation 스코어 업데이트
        
        if batch_idx % 100 == 0 :
            img_batch, label_grid_batch = batch
            pred_multiscales = self.forward(img_batch)
            
            BATCH_SIZE = img_batch.shape[0]

            losses = []
            mean_loss = 0

            # calculate loss
            for scale_idx in range(len(self.multiscales)) :
                # pred = pred_multiscales[scale_idx].contiguous().reshape(-1,  self.numbox, self.multiscales[scale_idx], self.multiscales[scale_idx],  (5 + self.num_classes))
                loss = self.yolo_loss(predictions = pred_multiscales[scale_idx], target = label_grid_batch[scale_idx], anchors = self.anchorbox[scale_idx])

                losses.append(loss)      

            mean_loss = sum(losses) / len(losses)
            self.log("val_loss", mean_loss)

            pred_bboxes_list = self.convert_bboxes(pred_multiscales)
            pred_bboxes_after_nms = [nms(bboxes, confidence_threshold = 0.5, iou_threshold = 0.8) for bboxes in pred_bboxes_list]
            grid = self.make_visgrid(pred_bboxes_after_nms, img_batch)
            self.logger.experiment.add_image("bbox visualization", grid, self.global_step)

            gt_bboxes_list = self.convert_bboxes(label_grid_batch, is_preds = False)
            with torch.no_grad() :
                
                for idx in range(BATCH_SIZE) :
                    preds, targets = self.get_predgt(pred_bboxes_after_nms[idx], gt_bboxes_list[idx])
                    # update on mean average precision
                    self.mAP.update(preds = preds, target = targets)

                
    
    def on_validation_epoch_end(self):
        self.log_dict(self.mAP.compute())
        self.mAP.reset()

    


    def predict_step(self, img_batch, batch_idx):
        pred_multiscales = self.forward(img_batch)
        
        pred_bboxes_list = self.convert_bboxes(pred_multiscales)
        pred_bboxes_after_nms = [nms(bboxes, confidence_threshold = 0.5, iou_threshold = 0.8) for bboxes in pred_bboxes_list]

        bbox_visualization = []
        for idx, (img, pred_bboxes) in enumerate(zip(img_batch, pred_bboxes_after_nms)) :
            bbox_img = drawboxes(img, pred_bboxes, confidence_threshold = 0.5)
            bbox_visualization.append(bbox_img)
    
        return pred_bboxes_after_nms, bbox_visualization

