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

import sys
sys.path.append('../')
from global_utils import IoU, IoU_width_height

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

        self.multiscales = cfg.model.multiscales
        self.num_classes = cfg.model.num_classes
        self.anchorbox = cfg.model.anchorbox
        self.ignore_iou_thresh = cfg.model.ignore_iou_thresh

        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.predict_transform = predict_transform

    def setup(self, stage : str) :
        if stage == 'fit' :
            self.train_dataset = BDDDataset(self.train_imgdir, 
                                            self.train_jsonfile, 
                                            self.num_classes, 
                                            self.multiscales,
                                            self.anchorbox, 
                                            self.ignore_iou_thresh,
                                            is_train = True, transform = self.train_transform)

            self.test_dataset = BDDDataset(self.test_imgdir, 
                                            self.test_jsonfile, 
                                            self.num_classes,
                                            self.multiscales, 
                                            self.anchorbox, 
                                            self.ignore_iou_thresh,
                                            is_train = True, transform = self.test_transform)
        if stage == 'validate' or stage == 'test' :
            self.test_dataset = BDDDataset(self.test_imgdir, 
                                            self.test_jsonfile, 
                                            self.num_classes,
                                            self.multiscales, 
                                            self.anchorbox, 
                                            self.ignore_iou_thresh,
                                            is_train = True, transform = self.test_transform)
        if stage == 'predict' :
            self.predict_dataset = BDDDataset(self.predict_imgdir, 
                                            self.predict_jsonfile, 
                                            self.num_classes,
                                            self.multiscales, 
                                            self.anchorbox, 
                                            is_train = False, transform = self.predict_transform)

    def train_dataloader(self) :
        return DataLoader(self.train_dataset, 
                          shuffle = True,
                          batch_size = self.batch_size, 
                          num_workers = self.num_workers)
    def val_dataloader(self) :
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    def test_dataloader(self) :
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    def predict_dataloader(self) :
        return DataLoader(self.predict_dataset, batch_size = self.batch_size, num_workers = self.num_workers)



class BDDDataset(Dataset):
    def __init__(self, imgdir, jsonfile, num_classes, multiscales, anchorbox, ignore_iou_thresh = 0.5, is_train = True, transform=None):
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
            self.imgfiles = self.imgfiles[:20]

        self.num_classes = num_classes
        self.multiscales = multiscales
        self.anchorbox = torch.tensor(anchorbox[0] + anchorbox[1] + anchorbox[2])
        self.num_anchors = self.anchorbox.shape[0]

        self.ignore_iou_thresh = ignore_iou_thresh

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
            # convert bbox coordinates to yolo format in encode()
            label_grid_multiscale = self.encode(bboxes, labels, H, W)

            return image.float(), label_grid_multiscale

        else :
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            return image

    def __len__(self):
        return len(self.imgfiles)

    def encode(self, bboxes, labels, H, W):

        # for different 3 scales, each has 3 x gridsize x gridsize x (pr(obj), x,y,w,h,label)
        targets = [torch.zeros(self.num_anchors // len(self.multiscales), S, S, 6) for S in self.multiscales]
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



            iou_anchors = IoU_width_height(torch.tensor([w, h]), self.anchorbox)
            anchor_indices = iou_anchors.argsort(descending = True, dim = 0)
            has_anchor = [False] * 3 # each scale should have one anchor            
            for anchor_idx in anchor_indices :
                scale_idx = int(anchor_idx // len(self.multiscales))
                anchor_on_scale = int(anchor_idx % len(self.multiscales))
                S = self.multiscales[scale_idx]
                j, i = int(S * y_center), int(S * x_center)
                anchor_taken = targets[scale_idx][anchor_on_scale, j, i, 0]
                if not anchor_taken and not has_anchor[scale_idx] :
                    targets[scale_idx][anchor_on_scale, j, i, 0] = 1
                    x_cell, y_cell = S * x_center - i, S * y_center - j
                    width_cell, height_cell = (
                        w * S,
                        h * S
                    )
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, j, i, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, j, i, 5] = label
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh :
                    targets[scale_idx][anchor_on_scale, j, i, 0] = -1 # ignore
            
            
        return tuple(targets)


if __name__ == '__main__' :
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from utils import decode_labelgrid
    import sys
    sys.path.append('../')
    from global_utils import drawboxes
    import albumentations as A
    import albumentations.pytorch as pytorch

    initialize(config_path="../config", job_name="test datamodule")
    cfg = compose(config_name="yolov2")
    print(cfg)

    train_transform = A.Compose(
    [   
        # A.Normalize(), # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        A.geometric.resize.RandomScale(scale_limit=cfg.aug.scale_limit),
        A.geometric.transforms.Affine(translate_percent = [cfg.aug.translation, cfg.aug.translation]), 
        A.geometric.resize.SmallestMaxSize(max_size=cfg.model.img_size),
        # A.transforms.ColorJitter(brightness = cfg.aug.brightness, saturation = cfg.aug.saturation), 
        A.RandomCrop(width=cfg.model.img_size, height=cfg.model.img_size, always_apply=True, p=1.0),
        A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None), 
        pytorch.transforms.ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["label"], min_visibility=0.8 # bounding box will be changed into yolo format after the encoding
    ),)

    datamodule = BDDDataModule(cfg, train_transform = train_transform)

    datamodule.setup('fit')

    train_dataloader = datamodule.train_dataloader()

    img_batch, label_batch = next(iter(train_dataloader))

    def write_txt(bboxes, fname) :
        with open(fname, 'w') as f :
            for bbox in bboxes :
                bbox = list(map(float, bbox))
                f.write('\t'.join(map(str, bbox)) + '\n')



    for idx, (img, label) in enumerate(zip(img_batch, label_batch)) :
        bboxes = decode_labelgrid(label, cfg.model.anchorbox, cfg.model.num_classes)

        bbox_img = drawboxes(img, bboxes)
        plt.imsave(f'experiment/check_img{idx}.jpg', bbox_img)
        fname = f'experiment/check_img{idx}.txt'
        write_txt(bboxes, fname)




    