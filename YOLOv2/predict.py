import hydra
from omegaconf import DictConfig

from ast import Num
from model import Yolov2
from data import BDDDataModule 


import torch
from PIL import Image
import matplotlib.pyplot as plt

import albumentations as A
import albumentations.pytorch as pytorch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path='../config', config_name='yolov2')
def predict(cfg: DictConfig) -> None :

    predict_transform = A.Compose(
            [
            A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
            A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
            A.Normalize(),
            pytorch.transforms.ToTensorV2(),
        ],
    )
    
    

    ## TODO : next time, grid_x and grid_y parameter in model weight should be clone() before expand()
    model = Yolov2(in_channels=3, 
                num_grid = cfg.model.num_grid, 
                anchorbox = cfg.model.anchorbox, 
                num_classes = cfg.model.num_classes)

    model.yolo_loss.grid_x = model.yolo_loss.grid_x.contiguous()
    model.yolo_loss.grid_y = model.yolo_loss.grid_y.contiguous()

    weight = torch.load(cfg.model.checkpoint)
    model.load_state_dict(weight['state_dict'])

    datamodule = BDDDataModule( cfg, 
                                predict_transform = predict_transform
                                )

    
    trainer = pl.Trainer()

    result_batch = trainer.predict(model = model,
                datamodule = datamodule,
                
                # when running in wsl, path should be changed accordingly
                # ckpt_path= cfg.model.checkpoint
                
    )

    for result in result_batch :
        bboxes, bbox_visualization = result

        for i, visualization in enumerate(bbox_visualization) :
            plt.imsave(f'val{i}.jpg', visualization.numpy())
    

if __name__ == '__main__' :

    predict()