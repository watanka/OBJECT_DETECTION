from ast import Num
from model import Yolov1
from data import BDDDataModule 


import torch
from torch.utils.data import DataLoader
from PIL import Image

import albumentations as A
import albumentations.pytorch as pytorch

import pytorch_lightning as pl

## TODO : config

model = Yolov1(in_channels=3, num_grid, num_boxes, num_classes)
datamodule = BDDDataModule(imgdir, jsonfile, num_grid, num_classes, numBox)

bbox_result = Trainer.predict(model = model,
             datamodule = datamodule,
             ckpt_path = 'best',
)