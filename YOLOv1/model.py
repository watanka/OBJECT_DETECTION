"""
Implementation of Yolo (v1) architecture
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

from loss import YOLOLoss
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
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(pl.LightningModule):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

        self.num_grid = kwargs["num_grid"]
        self.numbox = kwargs["numbox"]
        self.num_classes = kwargs["num_classes"]

        self.yolo_loss = YOLOLoss(
            lambda_coord=5,
            lambda_noobj=0.5,
            num_grid=self.num_grid,
            numbox=self.numbox,
        )

        self.mAP = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
            iou_thresholds=None,
        )  # iou_thresholds = None is same as [0.5, 0.05, 0.95]
        # self.mAP = MeanAveragePrecisionMetrics(gts, preds, iou_threshold_range, confidence_threshold)

        self.apply(self.initialize_weights)
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, num_grid, numbox, num_classes):

        S, B, C = num_grid, numbox, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        '''
        135 epochs 중에서 
        30 : 1e-3 (for the first epochs, number is not specified)
        45 : 1e-2 (out of 75 epochs, first epochs + current = 75)
        30 : 1e-3
        30 : 1e-4
        momentum 0.9, decay 5e-4
        '''
        def yolo_schedule(epoch) :
            if 0 <= epoch < 30 :
                lr = 1e-3
            elif 30 <= epoch < 75 :
                lr = 1e-2
            elif 75 <= epoch < 105 :
                lr = 1e-3
            elif 105 <= epoch < 135 :
                lr = 1e-4
            return lr


        scheduler = LambdaLR(optimizer, lr_lambda = yolo_schedule)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img_batch, label_grid = batch

        pred = self.forward(img_batch)
        pred = pred.contiguous().reshape(
            -1, self.num_grid, self.num_grid, (self.numbox * 5 + self.num_classes)
        )

        loss = self.yolo_loss(pred, label_grid)
        self.log("train_loss", loss)

        if batch_idx % 100 == 0 :
            pred = pred.detach().cpu()
            with torch.no_grad() :
                bboxes_batches = [nms(convert_labelgrid(p, numbox=self.numbox, num_classes=self.num_classes), confidence_threshold = 0.0, iou_threshold = 0.8) \
                                    for p in pred]

                bbox_visualization = []
                for img, bboxes in zip(img_batch, bboxes_batches) :
                    
                    bbox_visualization.append(torch.tensor(drawboxes(img, bboxes)))

                grid_result = torch.stack(bbox_visualization).permute(0,3,1,2)

                grid = torchvision.utils.make_grid(grid_result)
                self.logger.experiment.add_image("bbox visualization", grid, self.global_step)


        return loss

    def get_predgt(self, pred_bboxes_batch, gt_bboxes_batch) :
        '''
        bboxes to torchmetrics input format
        bboxes = [[conf_score, cx, cy, w, h, cls],...]
        '''
        pred_conf_batch, pred_coord_batch, pred_cls_batch = pred_bboxes_batch[..., 0], pred_bboxes_batch[..., 1:5], pred_bboxes_batch[..., -1]
        gt_obj_batch, gt_coord_batch, gt_cls_batch = gt_bboxes_batch[..., 0], gt_bboxes_batch[..., 1:5], gt_bboxes_batch[..., -1]

        preds = [
            dict(
                boxes = pred_coord_batch,
                scores = pred_conf_batch,
                labels = pred_cls_batch,
            )
        ]

        target = [
            dict(
                boxes = gt_coord_batch,
                labels = gt_cls_batch,
            )
        ]

        return preds, target


    def validation_step(self, batch, batch_idx):
        ### 100번마다 한 번씩만 validation 스코어 업데이트
        if batch_idx % 100 == 0 :
            img_batch, label_grid_batch = batch
            pred = self.forward(img_batch)
            pred = pred.reshape(
                -1, self.num_grid, self.num_grid, (self.numbox * 5 + self.num_classes)
            )

            loss = self.yolo_loss(pred, label_grid_batch)
            self.log("val_loss", loss, prog_bar= True)

            with torch.no_grad() :
                pred_bboxes_batch = [torch.tensor(nms(convert_labelgrid(p, numbox=self.numbox, num_classes=self.num_classes), confidence_threshold = 0.0, iou_threshold = 0.5)) \
                                    for p in pred]

                bbox_visualization = []
                for img, bboxes in zip(img_batch.detach(), pred_bboxes_batch) :

                    bbox_visualization.append(torch.tensor(drawboxes(img, bboxes.numpy())))

                grid_result = torch.stack(bbox_visualization).permute(0,3,1,2)

                grid = torchvision.utils.make_grid(grid_result)
                self.logger.experiment.add_image("bbox visualization", grid, self.global_step)


                gt_bboxes_batch = []
                for label_grid in label_grid_batch :
                    gt_bboxes_batch.append(decode_labelgrid(label_grid, numbox=self.numbox, num_classes=self.num_classes))
                gt_bboxes_batch = torch.tensor(gt_bboxes_batch)

                for pred_bboxes, gt_bboxes in zip(pred_bboxes_batch, gt_bboxes_batch) :
                    preds, target = self.get_predgt(pred_bboxes, gt_bboxes)
                    self.mAP.update(preds = preds, target = target)
    
    def on_validation_epoch_end(self):
        self.log_dict(self.mAP.compute(), prog_bar= True)
        self.mAP.reset()

    


    def predict_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred = pred.reshape(
            -1, self.num_grid, self.num_grid, (self.numbox * 5 + self.num_classes)
        )

        with torch.no_grad() :
            bboxes_batches = [nms(convert_labelgrid(p, numbox=self.numbox, num_classes=self.num_classes), confidence_threshold = 0.0, iou_threshold = 0.8) \
                                for p in pred]

            bbox_visualization = []
            for img, bboxes in zip(batch, bboxes_batches) :

                bbox_visualization.append(torch.tensor(drawboxes(img, bboxes)))
        
        return bboxes_batches, bbox_visualization