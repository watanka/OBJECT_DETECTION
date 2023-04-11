import hydra
from omegaconf import DictConfig

from ast import Num
from model import Yolov1
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

train_data_transform = A.Compose(
    [
        A.geometric.resize.SmallestMaxSize(max_size=446),
        # A.RandomCrop(width = 446, height = 446),
        A.RandomCrop(width=446, height=446, always_apply=True, p=1.0),
        A.PadIfNeeded(min_width=446, min_height=446, border_mode=None),
        pytorch.transforms.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["label"], min_visibility=0.8 # bounding box will be changed into yolo format after the encoding
    ),
)

## TODO : data_transform for inference
val_data_transform = A.Compose(
    [
        A.geometric.resize.LongestMaxSize(max_size=446),
        A.PadIfNeeded(min_width=446, min_height=446, border_mode=None),
        pytorch.transforms.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["label"], min_visibility=0.8
    ),
)


## TODO : organize configuration format
# cfg =

model = Yolov1(num_grid=7, num_boxes=2, num_classes=13).to(device)

train_dataset = BDDDataset(
    imgdir="../data/images/train",
    jsonfile="../data/label/det_train.json",
    num_grid=7,
    num_classes=13,
    numBox=2,
    transform=train_data_transform,
)

val_dataset = BDDDataset(
    imgdir="../data/images/val",
    jsonfile="../data/label/det_val.json",
    num_grid=7,
    num_classes=13,
    numBox=2,
    transform=val_data_transform,
)


train_dataloader = DataLoader(train_dataset, batch_size= 16, num_workers= 4)
val_dataloader = DataLoader(val_dataset, batch_size = 16, num_workers= 4)
## TODO : validation step

@hydra.main(config_path = 'config', config_name = 'config')
def train(cfg : DictConfig) -> None :

    train_dataloader = DataLoader(train_dataset, batch_size= 16, num_workers= 4)
    val_dataloader = DataLoader(val_dataset, batch_size = 16, num_workers= 4)




if __name__ == "__main__":
    
    ## logger
    tb_logger = TensorBoardLogger("tensorboard_log", name = 'yolov1')

    ckptCallback = ModelCheckpoint(dirpath = '', save_top_k = 2, monitor = 'val_loss')
    trainer = pl.Trainer(max_epochs=50, 
                         accelerator="gpu", 
                         logger = tb_logger,
                         callbacks= [ckptCallback],
                         )

    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        
        
    )

    trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path = 'last')
