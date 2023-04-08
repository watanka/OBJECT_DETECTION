import os
import json
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import albumentations as A

classes = ['traffic sign', 'traffic light', 'car', 'rider', 'motorcycle', 'pedestrian', 'bus', 'truck', 'bicycle', 'other vehicle', 'train', 'trailer', 'other person']

classes_dict = {c : i for i, c in enumerate(classes)}
label_dict = {num : clsname for clsname, num in classes_dict.items()}

class BDDDataset(Dataset) :
    def __init__(self, imgdir, jsonfile, num_grid, num_classes, numBox, transform = None ) :
        super().__init__()
        self.imgdir = imgdir
        # self.imgfiles = glob(os.path.join(self.imgdir, '*.jpg'))
        # self.imgfiles = sorted(self.imgfiles)
        
        with open(jsonfile, 'r') as f :
            json_infos = json.load(f)
        self.json_infos = sorted(json_infos, key = lambda x : x['name'])

        self.imgfiles = []
        for info in json_infos :
            self.imgfiles.append(os.path.join(self.imgdir, info['name']))

        ## make sure json information and imgfile are matched
        for info, imgfile in tqdm(zip(self.json_infos, self.imgfiles), desc = 'validate json and image matching...') :
            assert os.path.basename(imgfile) == info['name'], f"{imgfile}, {info['name']}"

        self.num_grid = num_grid
        self.num_classes = num_classes
        self.numBox = numBox

        self.transform = transform
        

    def __getitem__(self, idx) :
        imgfile = self.imgfiles[idx]
        json_info = self.json_infos[idx]

        json_info = json_info['labels'] if 'labels' in json_info.keys() else []

        image = plt.imread(imgfile) / 255.

        HEIGHT, WIDTH = image.shape[:2]
        
        bboxes = []
        labels = []
        for info in json_info :
            label = info['category']
            coord = info['box2d']
            # x, y = abs(float(coord['x'])), abs(float(coord['y']))
            # w, h = float(coord['w']), float(coord['h'])

            x1, y1 = coord['x1'], coord['y1']
            x2, y2 = coord['x2'], coord['y2']
            label = classes_dict[label]

            # bbox_albu = A.core.bbox_utils.convert_bbox_from_albumentations(np.array([x1, y1, x2, y2, label]), target_format='yolo', rows=HEIGHT, cols=WIDTH, check_validity= False)
            # print('bbox_albu : ', np.array(bbox_albu))
            # bbox_yolo = A.core.bbox_utils.convert_bbox_from_albumentations(np.array(bbox_albu), target_format='yolo', rows=HEIGHT, cols=WIDTH, check_validity=False)
            # print('bbox_yolo : ', bbox_yolo)
            
            
            # bboxes.append([x,y,w,h])
            bboxes.append([x1,y1,x2,y2])
            labels.append(label)

        if self.transform :
            transformed = self.transform(image = image, bboxes = bboxes, label = labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['label']
        

        H, W = image.shape[1:]
        label_grid = self.encode(bboxes, labels, H, W)

        return torch.tensor(image, dtype = torch.float32), torch.tensor(label_grid, dtype = torch.float32)
        

    def __len__(self) :
        return len(self.imgfiles)


    def encode(self, bboxes, labels, H, W) :
        label_grid = np.zeros((self.num_grid, self.num_grid, self.numBox * 5 + self.num_classes ), np.float32)
        grid_idxbox = np.zeros((self.num_grid, self.num_grid), np.uint8)
        label_box = np.full_like(grid_idxbox, -1) # grid cell당 하나의 class만 포함할 수 있다. 이미 grid안에 label값이 들어가있다면, continue.

        for bbox, label in zip(bboxes, labels) :

            x1, y1, x2, y2 = bbox

            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + w / 2) / W
            y_center = (y1 + h / 2) / H

            


            assert H == W, f'yolov1 takes only square image size. H = {H}, W = {W}'
            gridsize = 1 / self.num_grid

            grid_xidx = min(max(0, int(x_center // gridsize)), self.num_grid - 1) # there are cases center of bbox located at the endpoint
            grid_yidx = min(max(0, int(y_center // gridsize)), self.num_grid - 1) 

            ## normalize respect to the grid point
            grid_x0 = gridsize * grid_xidx
            grid_y0 = gridsize * grid_yidx

            normalized_x = (x_center - grid_x0) / gridsize
            normalized_y = (y_center - grid_y0) / gridsize

            boxnum = grid_idxbox[grid_yidx][grid_xidx]

            ## 하나의 grid cell은 하나의 class만 포함할 수 있다.
            if label_box[grid_yidx][grid_xidx] == -1 : # grid cell에 아무런 값이 들어가지 않은 경우, label을 넣어준다.
                label_box[grid_yidx][grid_xidx] = label

            elif label_box[grid_yidx][grid_xidx] == label : # label값이 이미 들어가있을 경우, 같은 클래스에 대한 값만 넣어준다.
                label_grid[grid_yidx, grid_xidx, self.numBox*5 + label] = 1.0
                if boxnum < self.numBox :
                    # 최대 self.numBox만큼만 넣는다.
                    # put into the grid
                    label_grid[grid_yidx, grid_xidx, boxnum*5 : boxnum*5 + 5 ] = [normalized_x, normalized_y, w, h, 1] 
                    grid_idxbox[grid_yidx][grid_xidx] += 1
                
            

        return label_grid