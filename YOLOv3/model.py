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
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import matplotlib.pyplot as plt
from tqdm import tqdm

from loss import YOLOv2loss
from utils import convert_labelgrid, decode_labelgrid
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
        if self.bn_act :
            return self.leaky(self.bn(self.conv(x)))
        else :
            return self.conv(x)

class ResidualBlock(nn.Module) :
    def __init__(self, channels, use_residual = True, num_repeats = 1)  :
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats) :
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kerne_size = 1),
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
        self.pred = nn.Sequential([
            CNNBlock(in_channels, 2*in_channels, kernel_size = 3, padding = 1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act = False, kerne_size = 1)
        ])
        self.num_classes = num_classes

    def forward(self, x) :
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2) # batch_size, numgrid, numgrid, 3, num_classes + 5
        )

class Yolov3(pl.LightningModule):
    def __init__(self, in_channels, multiscales, anchorbox, num_classes):
        super(Yolov2, self).__init__()
        self.register_buffer('anchorbox', torch.tensor(anchorbox))
        self.multiscales = multiscales
        self.numbox = len(self.anchorbox) // len(self.multiscales)
        self.num_classes = num_classes

        self.architecture = architecture_config
        self.in_channels = in_channels
        self.layers = self._create_conv_layers(self.architecture)

        
        self.yolo_loss = YOLOv2loss(
            lambda_coord=5,
            lambda_noobj=0.5,
            anchorbox=self.anchorbox,
            num_classes = self.num_classes,
            device = self.device
        )

        self.mAP = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
            iou_thresholds=None,
        )  # iou_thresholds = None is same as [0.5, 0.05, 0.95]
        # self.mAP = MeanAveragePrecisionMetrics(gts, preds, iou_threshold_range, confidence_threshold)

    

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers :
            if isinstance(layer, ScalePrediction) :
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8 :
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample) :
                x = torch.cat([x, route_connections[-1]], dim = 1) ##
                route_connections.pop()

        return outputs


        
    def _create_conv_layers(self, config):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config :
            if isinstance(module, tuple) :
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                            in_channels, 
                            out_channels,
                            kernel_size,
                            strid = stride,
                            padding = 1 if kernel_size == 3 else 0 
                            )
                        )
                in_channels = out_channels

            elif isinstance(module, list) :
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats = num_repeats))

            elif isinstance(module, str) :
                if module == 'S' :
                    layers += [
                        ResidualBlock(in_channels, use_residual= False, num_repeats= 1),
                        CNNBlock(in_channels, in_channels//2, kernel_size = 1),
                        ScalePrediction(in_channels // 2 , num_classes = num_classes)
                    ]
                    in_channels = in_channels//2
                elif module == 'U' :
                    layers.append(nn.Upsample(scale_factor = 2))
                    in_channels = in_channels * 3 
        return layers


        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay = 5e-4, momentum = 0.9)

        '''
        135 epochs 중에서 
        30 : 1e-3 (for the first epochs, number is not specified)
        45 : 1e-2 (out of 75 epochs, first epochs + current = 75)
        30 : 1e-3
        30 : 1e-4
        momentum 0.9, decay 5e-4
        '''
        def yolov2_schedule(epoch) :
            lr = 1e-3
            if epoch + 1 == 60 :
                lr *= 0.1
            elif epoch + 1 == 90 :
                lr *= 0.1
            return lr


        scheduler = LambdaLR(optimizer, lr_lambda = yolov2_schedule)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img_batch, label_grid = batch
        pred_multiscales = self.forward(img_batch)


        for scale_idx in range(len(self.multiscales)) :
            ## TODO 
            pred = pred_multiscales[scale_idx].contiguous().reshape(-1, self.num_grid, self.num_grid, self.numbox, (5 + self.num_classes))

            loss = self.yolo_loss(pred, label_grid)
            self.log("train_loss", loss)

            if batch_idx % 100 == 0 :

                with torch.no_grad() :

                    bbox_visualization = []

                    # visualization
                    for img, p in zip(img_batch, pred) :
                        bboxes = convert_labelgrid(p, anchorbox = self.anchorbox, num_classes=self.num_classes)
                        bboxes_after_nms = nms(bboxes, confidence_threshold = 0.3, iou_threshold = 0.8)
                        
                        bbox_img = drawboxes(img, bboxes_after_nms)
                        bbox_visualization.append(torch.tensor(bbox_img))

                    # hand over to to tensorboard
                    grid_result = torch.stack(bbox_visualization).permute(0,3,1,2)
                    grid = torchvision.utils.make_grid(grid_result)
                    self.logger.experiment.add_image("bbox visualization", grid, self.global_step)


        return loss

    def get_predgt(self, pred_bboxes, gt_bboxes) :
        '''
        bboxes to torchmetrics input format
        bboxes = [[conf_score, cx, cy, w, h, cls],...]
        '''

        # pred_conf, pred_coord, pred_cls = torch.tensor([]), torch.tensor([]), torch.tensor([])
        # gt_obj, gt_coord, gt_cls = torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # if len(pred_bboxes) > 0 :
        pred_bboxes = torch.tensor(pred_bboxes)
        pred_conf, pred_coord, pred_cls = pred_bboxes[..., 0], pred_bboxes[..., 1:5], pred_bboxes[..., -1]

        # if len(gt_bboxes) > 0 :
        gt_bboxes = torch.tensor(gt_bboxes)
        gt_obj, gt_coord, gt_cls = gt_bboxes[..., 0], gt_bboxes[..., 1:5], gt_bboxes[..., -1]

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
            pred = self.forward(img_batch)
            pred = pred.contiguous().reshape(-1, self.num_grid, self.num_grid, self.numbox, (5 + self.num_classes))
            

            loss = self.yolo_loss(pred, label_grid_batch)
            self.log("val_loss", loss)

            with torch.no_grad() :

                bbox_visualization = []
                # for loop by batch
                for img, gt_labelgrid, p in zip(img_batch, label_grid_batch, pred) :

                    pred_bboxes = convert_labelgrid(p, anchorbox = self.anchorbox, num_classes=self.num_classes)
                    pred_bboxes_after_nms = nms(pred_bboxes, confidence_threshold = 0.3, iou_threshold = 0.8)
                    
                    gt_bboxes = torch.tensor(decode_labelgrid(gt_labelgrid, self.anchorbox, self.num_classes))

                    # convert bbox format for mean average precision
                    preds, targets = self.get_predgt(pred_bboxes_after_nms, gt_bboxes)
                    # update on mean average precision
                    self.mAP.update(preds = preds, target = targets)


                    pred_bbox_img = drawboxes(img, pred_bboxes_after_nms)
                    bbox_visualization.append(torch.tensor(pred_bbox_img))

                # hand over to to tensorboard
                grid_result = torch.stack(bbox_visualization).permute(0,3,1,2)
                grid = torchvision.utils.make_grid(grid_result)
                self.logger.experiment.add_image("bbox visualization", grid, self.global_step)
    
    def on_validation_epoch_end(self):
        self.log_dict(self.mAP.compute())
        self.mAP.reset()

    


    def predict_step(self, img_batch, batch_idx):
        pred = self.forward(img_batch)
        pred = pred.contiguous().reshape(-1, self.num_grid, self.num_grid, self.numbox, (5 + self.num_classes))
        
        with torch.no_grad() :

            bbox_visualization = []
            bboxes_batches = []
            # visualization
            for p, img in zip(pred, img_batch) :
                bboxes = convert_labelgrid(p, anchorbox = self.anchorbox, num_classes=self.num_classes)
                bboxes_after_nms = nms(bboxes, confidence_threshold = 0.3, iou_threshold = 0.8)
                
                bbox_img = drawboxes(img, bboxes_after_nms)

                bboxes_batches.append(bboxes_after_nms)
                bbox_visualization.append(torch.tensor(bbox_img))
    
        return bboxes_batches, bbox_visualization