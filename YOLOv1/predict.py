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
    model = Yolov1(in_channels=3, 
                   num_grid = cfg.model.num_grid, 
                   numbox = cfg.model.numbox, 
                   num_classes = cfg.model.num_classes)

    datamodule = BDDDataModule(cfg.dataset.test.imgdir, 
                                cfg.dataset.test.jsonfile, 
                                cfg.model.num_grid, 
                                cfg.model.num_classes, 
                                cfg.model.numbox,
                                cfg.dataset.batch_size,
                                cfg.dataset.num_workers,
                                transform = None ## TODO : set with hydra 
                                )

    ckptCallback = ModelCheckpoint(dirpath = '', save_top_k = 2, monitor = 'val_loss')
    trainer = pl.Trainer(callbacks= [ckptCallback],)
    
    print(ckptCallback.best_model_path)
    bbox_result = trainer.test(model = model,
                datamodule = datamodule,
                ckpt_path = 'best',
    )


if __name__ == '__main__' :

    predict()