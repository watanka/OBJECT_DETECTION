import hydra

import os
import json
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

import pytorch_lightning as pl

classes = [
    "traffic sign",
    "traffic light",
    "car",
    "rider",
    "motorcycle",
    "pedestrian",
    "bus",
    "truck",
    "bicycle",
    "other vehicle",
    "train",
    "trailer",
    "other person",
]

classes_dict = {c: i for i, c in enumerate(classes)}
label_dict = {num: clsname for clsname, num in classes_dict.items()}


class BDDDataModule(pl.LightningDataModule) :
    def __init__(self, 
                 cfg, 
                 train_transform = None,
                 test_transform = None,
                 predict_transform = None
                 ) :
        super().__init__()
        self.cfg = cfg

        self.train_imgdir = cfg.dataset.train.imgdir
        self.train_jsonfile = cfg.dataset.train.jsonfile

        self.test_imgdir = cfg.dataset.test.imgdir
        self.test_jsonfile = cfg.dataset.test.jsonfile

        self.predict_imgdir = cfg.dataset.predict.imgdir
        self.predict_jsonfile = cfg.dataset.predict.jsonfile

        self.num_grid = cfg.model.num_grid
        self.num_classes = cfg.model.num_classes
        self.anchorbox = cfg.model.anchorbox

        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.predict_transform = predict_transform

    def setup(self, stage : str) :
        if stage == 'fit' :
            self.train_dataset = BDDDataset(self.train_imgdir, 
                                            self.train_jsonfile, 
                                            self.num_grid, 
                                            self.num_classes, 
                                            self.anchorbox, 
                                            is_train = True, transform = self.train_transform)

            self.test_dataset = BDDDataset(self.test_imgdir, 
                                            self.test_jsonfile, 
                                            self.num_grid, 
                                            self.num_classes, 
                                            self.anchorbox, 
                                            is_train = True, transform = self.test_transform)
        if stage == 'validate' or stage == 'test' :
            self.test_dataset = BDDDataset(self.test_imgdir, 
                                            self.test_jsonfile, 
                                            self.num_grid, 
                                            self.num_classes, 
                                            self.anchorbox, 
                                            is_train = True, transform = self.test_transform)
        if stage == 'predict' :
            self.predict_dataset = BDDDataset(self.predict_imgdir, 
                                            self.predict_jsonfile, 
                                            self.num_grid, 
                                            self.num_classes, 
                                            self.anchorbox, 
                                            is_train = False, transform = self.predict_transform)

    def train_dataloader(self) :
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    def val_dataloader(self) :
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    def test_dataloader(self) :
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    def predict_dataloader(self) :
        return DataLoader(self.predict_dataset, batch_size = self.batch_size, num_workers = self.num_workers)



class BDDDataset(Dataset):
    def __init__(self, imgdir, jsonfile, num_grid, num_classes, anchorbox, is_train = True, transform=None):
        super().__init__()
        self.imgdir = imgdir
        self.is_train = is_train
        if self.is_train :
            
            with open(jsonfile, "r") as f:
                json_infos = json.load(f)
            self.json_infos = sorted(json_infos, key=lambda x: x["name"])

            self.imgfiles = []
            for info in json_infos:
                self.imgfiles.append(os.path.join(self.imgdir, info["name"]))

            ## make sure json information and imgfile are matched
            for info, imgfile in tqdm(
                zip(self.json_infos, self.imgfiles),
                desc="validate json and image matching...",
            ):
                assert (
                    os.path.basename(imgfile) == info["name"]
                ), f"{imgfile}, {info['name']}"
        else :
            # predict, test => No Label
            self.imgfiles = glob(os.path.join(self.imgdir,'*.jpg')) + glob(os.path.join(self.imgdir,'*.png'))



        self.num_grid = num_grid
        self.num_classes = num_classes
        self.anchorbox = anchorbox

        self.transform = transform

    def __getitem__(self, idx):
        imgfile = self.imgfiles[idx]
        image = cv2.imread(imgfile) #/ 255.0

        if self.is_train :
            json_info = self.json_infos[idx]

            json_info = json_info["labels"] if "labels" in json_info.keys() else []

            

            HEIGHT, WIDTH = image.shape[:2]

            bboxes = []
            labels = []
            for info in json_info:
                label = info["category"]
                coord = info["box2d"]
                # x, y = abs(float(coord['x'])), abs(float(coord['y']))
                # w, h = float(coord['w']), float(coord['h'])

                x1, y1 = coord["x1"], coord["y1"]
                x2, y2 = coord["x2"], coord["y2"]
                label = classes_dict[label]

                bboxes.append([x1, y1, x2, y2])
                labels.append(label)

            if self.transform:
                transformed = self.transform(image=image, bboxes=bboxes, label=labels)
                image = transformed["image"]
                bboxes = transformed["bboxes"]
                labels = transformed["label"]

            H, W = image.shape[1:]
            label_grid = self.encode(bboxes, labels, H, W)

            return image.float(), label_grid

        else :
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            return image

    def __len__(self):
        return len(self.imgfiles)

    def encode(self, bboxes, labels, H, W):
        # yolov2 update : (num_grid, num_grid, anchorbox * (x,y,w,h,pr(obj), num_classes)). each anchor box predicts separate class
        # yolov2 update : separate array for each anchor box
        label_grid = np.zeros(
            (self.num_grid, self.num_grid, len(self.anchorbox), 5 + self.num_classes ), np.float32,)
        
        grid_idxbox = np.zeros((self.num_grid, self.num_grid), np.uint8)

        for bbox, label in zip(bboxes, labels):

            x1, y1, x2, y2 = bbox
            label = int(label)
            # convert pascal_voc to yolo format
            w = (x2 - x1)
            h = (y2 - y1)
            x_center = (x1 + w / 2) / W
            y_center = (y1 + h / 2) / H

            w /= W
            h /= H

            assert H == W, f"yolov2 takes only square image size. H = {H}, W = {W}"
            gridsize = 1 / self.num_grid

            # there are cases center of bbox located at the endpoint
            grid_xidx = min(max(0, int(x_center // gridsize)), self.num_grid - 1)  
            grid_yidx = min(max(0, int(y_center // gridsize)), self.num_grid - 1)

            # yolov2 normalize x,y respect to image size
            normalized_x = x_center / W
            normalized_y = y_center/ H

            boxnum = grid_idxbox[grid_yidx][grid_xidx]
            if boxnum < len(self.anchorbox) : # if total number of boxes exceeds len(anchorbox), then ignore.
                label_grid[grid_yidx, grid_xidx, boxnum, 5] = [1, normalized_x, normalized_y, w, h]
                label_grid[grid_yidx, grid_xidx, boxnum, 5 + label] = 1

                grid_idxbox[grid_yidx][grid_xidx] += 1
            
        return np.float32(label_grid)


if __name__ == '__main__' :
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    initialize(config_path="../config", job_name="test datamodule")
    cfg = compose(config_name="../config")

    datamodule = BDDDataModule(cfg)

    datamodule.setup('fit')

    train_dataloader = datamodule.train_dataloader()

    img_batch, label_batch = next(iter(train_dataloader))

    