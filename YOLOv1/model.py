
"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from loss import YOLOLoss
from utils import convert_prediction
import sys
sys.path.append('../')
from global_utils import visualize
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
    
        self.num_grid = kwargs['num_grid']
        self.num_boxes = kwargs['num_boxes']
        self.num_classes = kwargs['num_classes']
        
        self.yolo_loss = YOLOLoss(lambda_coord = 5, lambda_noobj = 0.5, num_grid = self.num_grid, numBox = self.num_boxes)

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
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
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

    def _create_fcs(self, num_grid, num_boxes, num_classes):
        
        S, B, C = num_grid, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

    def configure_optimizers(self) :
        optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer

    def training_step(self, batch, batch_idx) :
        img_batch, label_grid = batch
        
        pred = self.forward(img_batch)
        pred = pred.reshape(-1, self.num_grid, self.num_grid, (self.num_boxes * 5 + self.num_classes))

        loss = self.yolo_loss(pred, label_grid)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        img_batch, label_grid = batch
        pred = self.forward(img_batch)
        pred = pred.reshape(-1, self.num_grid, self.num_grid, (self.num_boxes * 5 + self.num_classes))

        loss = self.yolo_loss(pred, label_grid)
        self.log('val_loss', loss)

        for p,img in zip(pred, img_batch) :
            bboxes = convert_prediction(p, num_bboxes = self.num_boxes, num_classes = self.num_classes)
            visualized_img = visualize(img, bboxes)
            plt.imsave('val_imgs/validation_img.jpg', visualized_img)
            break

        

