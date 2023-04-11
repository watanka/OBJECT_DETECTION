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


@hydra.main(config_path='config', config_name='config')
def predict(cfg: DictConfig) -> None :
    model = Yolov1(in_channels=3, 
                   num_grid = cfg.model.num_grid, 
                   num_boxes = cfg.model.num_boxes, 
                   num_classes = cfg.model.num_classes)
    datamodule = BDDDataModule(cfg.dataset.test.imgdir, 
                                cfg.dataset.test.jsonfile, 
                                cfg.model.num_grid, 
                                cfg.model.num_classes, 
                                cfg.model.numBox,
                                transform = None ## TODO : set with hydra 
                                )

    bbox_result = Trainer.predict(model = model,
                datamodule = datamodule,
                ckpt_path = 'best',
    )


if __name__ == '__main__' :

    predict()