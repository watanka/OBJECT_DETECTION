from ast import Num
from model import Yolov1
from data import BDDDataset


import torch
from torch.utils.data import DataLoader
from PIL import Image

import albumentations as A
import albumentations.pytorch as pytorch

import pytorch_lightning as pl

Trainer.test(model = ,
             dataloaders = ,
             ckpt_path = 'best',
)