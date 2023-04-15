import hydra
from omegaconf import DictConfig
import logging
import os

from model import Yolov2
from data import BDDDataset, BDDDataModule


import torch
from torch.utils.data import DataLoader
from PIL import Image

import albumentations as A
import albumentations.pytorch as pytorch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

log = logging.getLogger(__name__)




@hydra.main(config_path = '../config', config_name = 'yolov2')
def train(cfg : DictConfig) -> None :

    train_transform = A.Compose(
    [   A.Normalize(), # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        A.geometric.resize.RandomScale(scale_limit=cfg.aug.scale_limit),
        A.geometric.transforms.Affine(translate_percent = [cfg.aug.translation, cfg.aug.translation]), 
        A.geometric.resize.SmallestMaxSize(max_size=cfg.model.img_size),
        A.transforms.ColorJitter(brightness = cfg.aug.brightness, saturation = cfg.aug.saturation), 
        A.RandomCrop(width=cfg.model.img_size, height=cfg.model.img_size, always_apply=True, p=1.0),
        A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None), 
        pytorch.transforms.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["label"], min_visibility=0.8 # bounding box will be changed into yolo format after the encoding
    ),
)

    test_transform = A.Compose(
        [
            A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
            A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
            A.Normalize(),
            pytorch.transforms.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["label"], min_visibility=0.8
        ),
    )

    predict_transform = A.Compose(
            [
            A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
            A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
            A.Normalize(),
            pytorch.transforms.ToTensorV2(),
        ],
    )

    datamodule = BDDDataModule(cfg,
                                train_transform = train_transform,
                                test_transform = test_transform,
                                predict_transform = predict_transform
                                )

    model = Yolov2(in_channels = 3, num_grid= cfg.model.num_grid, anchorbox=cfg.model.anchorbox, num_classes=cfg.model.num_classes)            
    
    ## logger
    tb_logger = TensorBoardLogger(save_dir = os.path.join(os.getcwd(), 'tensorboard/') , name = 'yolov2', version = '0')
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

@hydra.main(config_path = '../config', config_name = 'yolov2')
def validate(cfg : DictConfig) -> None :
    train_transform = A.Compose(
    [   A.Normalize(), # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        A.geometric.resize.RandomScale(scale_limit=cfg.aug.scale_limit),
        A.geometric.transforms.Affine(translate_percent = [cfg.aug.translation, cfg.aug.translation]), 
        A.geometric.resize.SmallestMaxSize(max_size=cfg.model.img_size),
        A.transforms.ColorJitter(brightness = cfg.aug.brightness, saturation = cfg.aug.saturation), 
        A.RandomCrop(width=cfg.model.img_size, height=cfg.model.img_size, always_apply=True, p=1.0),
        A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None), 
        pytorch.transforms.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["label"], min_visibility=0.8 # bounding box will be changed into yolo format after the encoding
    ),
)

    test_transform = A.Compose(
        [
            A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
            A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
            A.Normalize(),
            pytorch.transforms.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["label"], min_visibility=0.8
        ),
    )

    predict_transform = A.Compose(
            [
            A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
            A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
            A.Normalize(),
            pytorch.transforms.ToTensorV2(),
        ],
    )    

    datamodule = BDDDataModule( cfg,
                                train_transform = train_transform,
                                test_transform = test_transform,
                                predict_transform = predict_transform
                                )

    model = Yolov1(in_channels = 3, num_grid= cfg.model.num_grid, numbox=cfg.model.numbox, num_classes=cfg.model.num_classes)            
    
    ## logger
    tb_logger = TensorBoardLogger(save_dir = os.path.join(os.getcwd(), 'tensorboard/') , name = 'yolov1')
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

    trainer.validate(
        model=model, datamodule= datamodule)


if __name__ == "__main__":
    train()    