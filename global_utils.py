import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np



def IoU(pred, gt) :
    '''Intersection over Union'''

    pred_x_center = pred[..., 0:1]
    pred_y_center = pred[..., 1:2]
    pred_w = pred[..., 2:3]
    pred_h = pred[..., 3:4]

    gt_x_center= gt[..., 0:1]
    gt_y_center = gt[..., 1:2]
    gt_w = gt[..., 2:3]
    gt_h = gt[..., 3:4]


    pred_x1 = pred_x_center - pred_w / 2
    pred_y1 = pred_y_center - pred_h / 2

    pred_x2 = pred_x_center + pred_w / 2
    pred_y2 = pred_y_center + pred_h / 2



    gt_x1 = gt_x_center - gt_w / 2
    gt_y1 = gt_y_center - gt_h / 2

    gt_x2 = gt_x_center + gt_w / 2
    gt_y2 = gt_y_center + gt_h / 2

    x1 = torch.max(pred_x1, gt_x1)
    y1 = torch.max(pred_y1, gt_y1)

    x2 = torch.min(pred_x2, gt_x2)
    y2 = torch.min(pred_y2, gt_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    total_area = ((pred_x2 - pred_x1) * (pred_y2 - pred_y1)) + ((gt_x2 - gt_x1) * (gt_y2 - gt_y1))

    return intersection / (total_area - intersection + 1e-6) # add buffer

def MAP(pred, gt, iou_threshold, confidence_threshold) :
    '''
    Calculate Mean Average Precision
    '''

    pass


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # white

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=1):
    """Visualizes a single bounding box on the image"""
    img_h, img_w = img.shape[:2]

    x, y, w, h = bbox
    x_min, x_max, y_min, y_max = x - w/2, x + w/2, y - h/2, y + h/2
    x_min = int(img_w * x_min)
    x_max = int(img_w * x_max)
    y_min = int(img_h * y_min)
    y_max = int(img_h * y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids):
    if type(image) == torch.Tensor :
        image = image.permute(1,2,0).detach().cpu().numpy()
    else :
        image = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

def visualize_gridbbox(img, label_grid, numBox = 2, color = BOX_COLOR, thickness = 1) :
    H, W = img.shape[:2]
    
    if type(img) == torch.Tensor :
        copy_img = img.permute(1,2,0).detach().cpu().numpy().copy()
    else :
        copy_img = img.copy()
    assert H == W, 'image size should be same.'
    num_grid, _, info_length = label_grid.shape

    gridsize = H // num_grid

    for yidx in range(num_grid) :
        for xidx in range(num_grid) :
            
            for box_idx in range(numBox) : 

                x0, y0 = xidx, yidx
                x_center, y_center, w, h = label_grid[yidx, xidx, box_idx * 5 : box_idx * 5 + 4]
                x_center = int((x_center + x0) * gridsize)
                y_center = int((y_center + y0) * gridsize)

                xmin = int(x_center - w * W / 2)
                ymin = int(y_center - h * H / 2)
                xmax = int(x_center + w * W / 2)
                ymax = int(y_center + h * H / 2)

                cv2.rectangle(copy_img, (xmin, ymin), (xmax, ymax), color = color, thickness = thickness)
    
    return copy_img
    

    
    
def nms(bboxes, threshold, iou_threshold) :
    '''
    bboxes : [[x,y,w,h,c], ]. c = confidence score
    threshold : confidence thresholds
    iou_threshold : if boxes overlap over iou_threshold, eliminate from the candidates
    '''
    bboxes = [box for box in bboxes if box[4] > threshold]
    # sort by highest confidence
    bboxes = sorted(bboxes, key = lambda x : x[4]) 

    bboxes_after_nms = []
    while bboxes :
        chosen_box = bboxes.pop()

        bboxes = [
            box
            for box in bboxes
            if box[4] != chosen_box[4] \
                or IoU(np.array(box), np.array(chosen_box)) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



