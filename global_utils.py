import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


def IoU_width_height(box1, box2) :
    '''
    calculate IoU as if the center point is fixed.
    used to calculate corresponding anchor box to gt.
    '''
    intersection = torch.min(box1[...,0], box2[...,1]) * torch.min(box1[..., 1], box2[...,1])
    union = box1[...,0] * box1[...,1] + box2[...,0] * box2[...,1] - intersection
    
    return intersection / (union + 1e-9)

def IoU(box1, box2) :
    # box = [x,y,w,h]

    
    box1_w = box1[..., 2:3]
    box1_h = box1[..., 3:4]
    box2_w = box2[..., 2:3]
    box2_h = box2[..., 3:4]

    box1_area = box1_w * box1_h
    box2_area = box2_w * box2_h

    box1_xmin = box1[..., 0:1] - box1_w / 2
    box1_ymin = box1[..., 1:2] - box1_h / 2
    box1_xmax = box1[..., 0:1] + box1_w / 2
    box1_ymax = box1[..., 1:2] + box1_w / 2

    box2_xmin = box2[..., 0:1] - box2_w / 2
    box2_ymin = box2[..., 1:2] - box2_h / 2
    box2_xmax = box2[..., 0:1] + box2_w / 2
    box2_ymax = box2[..., 1:2] + box2_w / 2
    
    box1_topleft = torch.cat([box1_xmin, box1_ymin], axis = -1)
    box2_topleft = torch.cat([box2_xmin, box2_ymin], axis = -1)

    box1_bottomright = torch.cat([box1_xmax, box1_ymax], axis = -1)
    box2_bottomright = torch.cat([box2_xmax, box2_ymax], axis = -1)

    top_left = torch.max(box1_topleft, box2_topleft)
    bottom_right = torch.min(box1_bottomright, box2_bottomright)



    area_inter = torch.prod(
        torch.clip(bottom_right - top_left, min = 0 , max = None), -1).unsqueeze(-1)
    


    return area_inter / (box1_area + box2_area - area_inter + 1e-9)



BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # white
classes = ['traffic sign', 'traffic light', 'car', 'rider', 'motorcycle', 'pedestrian', 'bus', 'truck', 'bicycle', 'other vehicle', 'train', 'trailer', 'other person']

classes_dict = {c : i for i, c in enumerate(classes)}
label_dict = {num : clsname for clsname, num in classes_dict.items()}

def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=1):
    """
    bbox = [pred_cls, x, y, w, h, confidence_score]
    Visualizes a single bounding box on the image
    
    """
    img_h, img_w = img.shape[:2]

    confidence_score, x, y, w, h, pred_cls = map(float, bbox)
    pred_cls = int(pred_cls)
    
    x_min, x_max, y_min, y_max = x - w/2, x + w/2, y - h/2, y + h/2
    x_min = int(img_w * x_min)
    x_max = int(img_w * x_max)
    y_min = int(img_h * y_min)
    y_max = int(img_h * y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(label_dict[pred_cls], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=label_dict[pred_cls],
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def drawboxes(img, bboxes):
    if type(img) == torch.Tensor :
        img = img.permute(1,2,0).detach().cpu().numpy()
        img = np.array(img*255., dtype = np.uint8).copy()
    else :
        img = img.copy()

    for bbox in bboxes:
        img = visualize_bbox(img, bbox)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)
    return img

def visualize_gridbbox(img, label_grid, numBox = 2, color = BOX_COLOR, thickness = 1) :
    if type(img) == torch.Tensor :
        copy_img = img.permute(1,2,0).detach().cpu().numpy().copy()
        H, W = copy_img.shape[:2]
    else :
        copy_img = img.copy()
        H, W = copy_img.shape[:2]
    assert H == W, 'image size should be same.'
    num_grid, _, info_length = label_grid.shape

    gridsize = H // num_grid

    for yidx in range(num_grid) :
        for xidx in range(num_grid) :
            
            for box_idx in range(numBox) : 

                x0, y0 = xidx, yidx
                x_center, y_center, w, h = label_grid[yidx, xidx, box_idx * 5 + 1 : (box_idx+1) * 5]
                x_center = int((x_center + x0) * gridsize)
                y_center = int((y_center + y0) * gridsize)

                xmin = int(x_center - w * W / 2)
                ymin = int(y_center - h * H / 2)
                xmax = int(x_center + w * W / 2)
                ymax = int(y_center + h * H / 2)

                cv2.rectangle(copy_img, (xmin, ymin), (xmax, ymax), color = color, thickness = thickness)
    
    return copy_img
    

    
    
# def nms(bboxes, threshold, iou_threshold) :
#     '''
#     bboxes : [[c, x,y,w,h,class], ]. c = confidence score
#     threshold : confidence thresholds
#     iou_threshold : if boxes overlap over iou_threshold, eliminate from the candidates
#     '''

#     labels = set([box[-1] for box in bboxes])

#     total_bboxes_after_nms = []

#     for label in labels :

#         bboxes = [box for box in bboxes if box[0] > threshold and box[-1] == label]
#         # sort by highest confidence
#         bboxes = sorted(bboxes, key = lambda x : x[0]) 

#         bboxes_after_nms = []
#         while bboxes :
#             chosen_box = bboxes.pop()

#             bboxes = [
#                 box
#                 for box in bboxes
#                 if box[0] != chosen_box[0] \
#                     or IoU(np.array(box[1:5]), np.array(chosen_box[1:5])) < iou_threshold
#             ]

#             bboxes_after_nms.append(chosen_box)
#         total_bboxes_after_nms.extend(bboxes_after_nms)

#     return np.array(total_bboxes_after_nms)

def nms(predictions, confidence_threshold: float , iou_threshold: float) :
    '''
    vectorize nms
    reference : https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/
    '''
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:,0].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, 1:5]
    categories = predictions[:, -1]
    ious = IoU(torch.tensor(boxes), torch.tensor(boxes)).detach().cpu().numpy()
    ious = ious - np.eye(rows)
    keep = predictions[:, 0] > confidence_threshold # np.ones(rows, dtype = bool)

    for index, (iou, category) in enumerate(zip(ious, categories)) :
        if not keep[index] :
            continue
        
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return predictions[keep[sort_index.argsort()]]


## MAP

class MeanAveragePrecisionMetrics :
    def __init__(self, num_classes, iou_threshold_range, confidence_threshold) :
        ''' 
        reference : https://github.com/rafaelpadilla/Object-Detection-Metrics
        usage : validation, evaluation. 
        predictions and gts will be handed by reading gtfile directly, since dataloader limits the number of gt bounding box
        '''

        
        '''
        ## TODO : [conf, x, y, w, h, cls]
        gts, preds = [[[class, x, y, w, h, c],...], ...] # imgs x bboxes

        
        num_classes : updated after gt and preds come in. should be collected from gt.
        iou_threshold_range : (min_threshold, interval, max_threshold). e.g) IoU(0.6, 0.1, 0.9) = [0.6, 0.7, 0.8, 0.9]
        confidence_threshold : predicted bounding boxes are filtered by confidence threshold
        '''
        
        self.num_classes = num_classes
        # convert iou_threshold_range into list
        min_threshold, interval, max_threshold  = iou_threshold_range
        self.iou_threshold_range = np.linspace(min_threshold, max_threshold, num= int((max_threshold - min_threshold)//interval + 1))
        
        self.confidence_threshold = confidence_threshold
        self.iou_table_per_img = {imgidx : None for imgidx in range(len(gts))}

        # self.total_iou_table = {clslabel : {iou_threshold : None \
        #                                         for iou_threshold in self.iou_threshold_range} \
        #                                             for clslabel in range(num_classes)}

        self.TOTAL_TP = [{iou_threshold : 0 for iou_threshold in self.iou_threshold_range} for _ in range(self.num_classes)]
        self.TOTAL_FP = [{iou_threshold : 0 for iou_threshold in self.iou_threshold_range} for _ in range(self.num_classes)]
        self.TOTAL_FN = [{iou_threshold : 0 for iou_threshold in self.iou_threshold_range} for _ in range(self.num_classes)]

        self.total_statistics = []


    def calculate_PR(self, imgidx, pred, gt, iou_threshold) :
        '''
        calculate precision and recall per classes
        Recall = TP / (TP+FN)
        Precision = TP / (TP + FP)
        1. match preds and gts using IoU
        2. matched preds will be TP, and remaining unmatched preds will be FP, and unmatched gt are FN.
        '''

        # bboxes = [[class,x,y,w,h,c],...]. c = confidence_score. filter predicted bboxes by confidence_threshold.
        
        pred = np.array([p for p in pred if p[-1] > self.confidence_threshold])
        
        
        # 행 : pred, 열 : gt
        if self.iou_table_per_img[imgidx] is not None :
            iou_table = self.iou_table_per_img[imgidx]
        else : 
            iou_table = torch.zeros((len(pred), len(gt)))
            for j, pred_bbox in enumerate(pred) :
                for i, gt_bbox in enumerate(gt) :
                    iou_table[j][i] = IoU(pred_bbox[..., 1:], gt_bbox[..., 1:])
            self.iou_table_per_img[imgidx] = iou_table
        # if there are more than one prediction box matched with on gt box, then we choose the predicition with the highest IoU as TP, and treat other matches as FP.     
        filtered_iou_table = torch.zeros_like(iou_table)
        filtered_iou_table[torch.argmax(iou_table, axis = 0), torch.arange(iou_table.shape[1])] = torch.max(iou_table, axis = 0)[0] # this will leave only one highest IoU per gts
        
        result = filtered_iou_table > iou_threshold

        TP = result.any(axis = 1).sum()
        FP = len(pred) - TP
        FN = len(gt) - result.any(axis = 0).sum()

        return TP.item(), FP.item(), FN.item()

    def calculate_average_precision(self, precision, recall) :

        precision = list(precision)[:] # [:] = copy list
        recall = list(recall)[:]

        mean_precision = [0] + precision + [0] 
        mean_recall = [0] + recall + [0]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
        """
        for i in range(len(mean_precision)-2, -1, -1) :
            mean_precision[i] = max(mean_precision[i], mean_precision[i+1])
        
        """
        This part creates a list of indexes where the recall changes
        """
        i_list = []
        for i in range(1, len(mean_recall)) :
            if mean_recall[i] != mean_recall[i-1] :
                i_list.append(i)
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
        """
        average_precision = 0.0
        for i in i_list :
            average_precision += ((mean_recall[i] - mean_recall[i-1]) * mean_precision[i])

        # average_precision = torch.mean(torch.trapz(torch.tensor(precision), torch.tensor(recall))).item()
        return average_precision


    def calculate(self, preds, gts) :
        ## TODO : add typing of variables
        '''
        preds : list of numpy array. []
        gts 
        iou_threshold_range : (minimum_threshold, maximum_threshold, interval)
        '''

        cnt_cls = set()
        for gts_per_img in gts :
            for gt_bbox in gts_per_img :
                cnt_cls.add(int(gt_bbox[0]))
        classes = list(cnt_cls)

        for imgidx, (pred_by_img, gt_by_img) in enumerate(zip(preds, gts)) :
            for cls_label in classes :
                cls_preds = pred_by_img[pred_by_img[..., 0] == cls_label]
                cls_gts = gt_by_img[gt_by_img[..., 0] == cls_label]

                # assert (cls_preds.shape == cls_gts.shape) and (cls_preds.ndim == cls_gts.ndim == 2 and cls_preds.shape[-1] == cls_gts.shape[-1] == 5), \
                #             'pred and gt shape = (# of bboxes over images, len([x,y,w,h,c]) )'

                for iou_threshold in self.iou_threshold_range :
                    TP, FP, FN = self.calculate_PR(imgidx, cls_preds, cls_gts, iou_threshold)
                    self.TOTAL_TP[cls_label][iou_threshold] += TP
                    self.TOTAL_FP[cls_label][iou_threshold] += FP
                    self.TOTAL_FN[cls_label][iou_threshold] += FN

        # calculate Precision and Recall

        for cls_label in classes :
            for iou_threshold in self.iou_threshold_range :
                ## round by 3 decimal places
                precision = round(self.TOTAL_TP[cls_label][iou_threshold] / (self.TOTAL_TP[cls_label][iou_threshold] + self.TOTAL_FP[cls_label][iou_threshold] + 1e-6), 3) # add 1e-6 to prevent divisionbyzero
                recall = round(self.TOTAL_TP[cls_label][iou_threshold] / (self.TOTAL_TP[cls_label][iou_threshold] + self.TOTAL_FN[cls_label][iou_threshold] + 1e-6), 3)

                self.total_statistics.append([cls_label, iou_threshold, precision, recall])

        self.total_statistics = np.array(self.total_statistics)

        # mean average precision = sum(avg_cls_precision) / num_classes. avg_cls_precision = sum of cls_precisions in different recalls / 
        mean_average_precision = 0
        
        for cls_label in classes :
            class_statistics = self.total_statistics[self.total_statistics[..., 0] == cls_label][...,2:4].tolist() # ious x [precision, recall]
            class_statistics = sorted(class_statistics, key = lambda x : x[1])
            precision, recall = zip(*class_statistics)
            
            average_precision = self.calculate_average_precision(precision, recall)
            mean_average_precision += average_precision
            

        mean_average_precision /= len(classes)

        return mean_average_precision

    