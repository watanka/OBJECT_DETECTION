import hydra
from omegaconf import DictConfig
import logging
import os

from model import Yolov4
from data import BDDDataset, BDDDataModule


import torch
from torch.utils.data import DataLoader
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append('../')
from transform import DataAug


@hydra.main(config_path = '../config', config_name = 'yolov4')
def train(cfg : DictConfig) -> None :
    data_aug = DataAug(cfg)

    datamodule = BDDDataModule(cfg,
                                train_transform = data_aug.train_transform,
                                test_transform = data_aug.test_transform,
                                predict_transform = data_aug.predict_transform
                                )

    model = Yolov4(in_channels_list = cfg.model.backbone_channels, multiscales= cfg.model.multiscales, anchorbox=cfg.model.anchorbox, num_classes=cfg.model.num_classes)            
    
    ## logger
    tb_logger = TensorBoardLogger(save_dir = os.path.join(os.getcwd(), 'tensorboard/') , name = 'yolov4', version = '0')
    # print(f'Model weight will be saved with tensorboard logger {tb_logger.save_dir}.')
    ckptCallback = ModelCheckpoint(dirpath = tb_logger.save_dir, 
                                   filename = cfg.schedule.savefile_format,
                                   save_top_k = 2, 
                                   monitor = 'val_loss')
    trainer = pl.Trainer(max_epochs= cfg.schedule.max_epochs, 
                         accelerator="gpu", 
                         logger = tb_logger,
                         callbacks= [ckptCallback],
                         )

    trainer.fit(
        model=model, datamodule= datamodule)


if __name__ == "__main__":
    train()    
