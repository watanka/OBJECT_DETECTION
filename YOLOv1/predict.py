import hydra
from omegaconf import DictConfig

from ast import Num
from model import Yolov1
from data import BDDDataModule 


import torch
from torch.utils.data import DataLoader
from PIL import Image

import albumentations as A
import albumentations.pytorch as pytorch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path='../config', config_name='config')
def predict(cfg: DictConfig) -> None :

    predict_transform = A.Compose(
            [
            A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
            A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
            A.Normalize(),
            pytorch.transforms.ToTensorV2(),
        ],
    )    
    model = Yolov1(in_channels=3, 
                   num_grid = cfg.model.num_grid, 
                   numbox = cfg.model.numbox, 
                   num_classes = cfg.model.num_classes)

    datamodule = BDDDataModule( cfg, 
                                predict_transform = predict_transform ## TODO : set with hydra 
                                )

    ckptCallback = ModelCheckpoint(dirpath = '', save_top_k = 2, monitor = 'val_loss')
    trainer = pl.Trainer(callbacks= [ckptCallback],)

    result_batch = trainer.predict(model = model,
                datamodule = datamodule,
                # when running in wsl, path should be changed accordingly
                ckpt_path = '/mnt/c/Users/user/Desktop/projects/object_detection/YOLOv1/outputs/2023-04-12/14-18-40/tensorboard/yolov1-epoch=07-val_lossval_loss=0.00.ckpt',
    )

    for result in result_batch :
        bboxes, bbox_visualization = result
        print(len(bboxes))
        print(bbox_visualization[0].shape)

    

if __name__ == '__main__' :

    predict()