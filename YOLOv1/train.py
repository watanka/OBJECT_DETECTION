from ast import Num
from model import Yolov1
from data import BDDDataset


import torch
from torch.utils.data import DataLoader
from PIL import Image

import albumentations as A
import albumentations.pytorch as pytorch

import pytorch_lightning as pl

device = torch.device('gpu' if torch.cuda.is_available() \
                        else 'mps' if torch.backends.mps.is_available() \
                        else 'cpu'
                        )

train_data_transform = A.Compose([
    A.geometric.resize.SmallestMaxSize(max_size = 446),
    # A.RandomCrop(width = 446, height = 446),
    
    A.RandomCrop(width = 446, height = 446, always_apply = True, p = 1.0),
    
    A.PadIfNeeded(min_width = 446, min_height = 446, border_mode=None),
    pytorch.transforms.ToTensorV2(),
], bbox_params = A.BboxParams(format = 'yolo', label_fields = ['label'], min_visibility = 0.8))

## TODO : data_transform for inference


## TODO : organize configuration format
# cfg = 

model_dict = dict(num_grid = 7, num_boxes = 2, num_classes = 13)

model = Yolov1(num_grid = 7, num_boxes = 2, num_classes = 13).to(device)

dataset = BDDDataset(imgdir = '../data/images/val', 
           jsonfile = '../data/label/yolo_det_val.json', 
           num_grid = 7, 
           num_classes = 13, 
           numBox = 2 , 
           transform = train_data_transform )

train_dataloader = DataLoader(dataset)

## TODO : validation step


if __name__ == '__main__' :

    trainer = pl.Trainer(max_epochs = 1, accelerator = 'gpu')
    trainer.fit(model = model, train_dataloaders = train_dataloader)