import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import json
import random

from global_utils import IoU
import torch


## convert labelgrid into (confidence_scores, bboxes, labels)
def convert_labelgrid(label_grid, anchorbox, num_classes):
    """
    operations are grid units.

    MAKE SURE label_grid is NOT BATCHWISE.

    label_grid : output of the model. num_grid x num_grid x len(anchorbox) x (5 + num_classes)
    numbox : maximum number of boxes per grid cell
    num_classes : # of classes
    """

    num_grid, num_grid, _, seq_length = label_grid.shape

    # get class index with highest probabilities
    classlist_grid = label_grid[..., 5 :]
    max_probability_class_grid = torch.argmax(classlist_grid, -1)

    # select coordinates and confidence scores with highest probability per grid cell
    coords_grid = label_grid[..., 1 : 5]

    # select only the confidence score indexes
    confidence_grid = label_grid[..., 0]

    ## convert ratio respect to the image size => add y0, x0 coordinates. (y0, x0) differ by grid cell.
    gridsize = 1 / num_grid
    bboxes = []
    for j in range(num_grid):
        for i in range(num_grid):
            for boxidx in range(len(anchorbox)) :
                
                # cy, cx is offset from the top left corner of the image
                cy = gridsize * j
                cx = gridsize * i
                anchor_h, anchor_w = anchorbox[boxidx].detach().cpu().numpy()
                confidence_score = confidence_grid[j][i].detach().cpu().numpy()
                raw_x, raw_y, raw_w, raw_h = coords_grid[j][i].detach().cpu().numpy()


                y = torch.sigmoid(raw_y) + cy
                x = torch.sigmoid(raw_x) + cx
                w = anchor_w * torch.exp(raw_w)
                h = anchor_h * torch.exp(raw_h)


                pred_cls = max_probability_class_grid[j][i].detach().cpu().numpy().tolist()

                bboxes.append([confidence_score, x, y, w, h, pred_cls])

    return bboxes

def decode_labelgrid(label_grid, anchorbox, num_classes) :
    '''For GT label conversion, separate from prediction conversion since gt label grid can have numBox true bboxes'''
    num_grid, num_grid, _, seq_length = label_grid.shape

    # get class index with highest probabilities
    classlist_grid = label_grid[..., 5 :]
    max_probability_class_grid = torch.argmax(classlist_grid, -1)

    # select coordinates and confidence scores with highest probability per grid cell
    coords_grid = label_grid[...,  : 5]

    ## convert ratio respect to the image size => add y0, x0 coordinates. (y0, x0) differ by grid cell.
    gridsize = 1 / num_grid
    bboxes = []
    for j in range(num_grid):
        for i in range(num_grid):
            for boxidx in range(len(anchorbox)) :
                confidence_score, x, y, w, h  = coords_grid[j][i][boxidx].detach().cpu().numpy().tolist() 
                label = max_probability_class_grid[j][i].detach().cpu().numpy().tolist()

                bboxes.append([confidence_score, x, y, w, h, label])

    return bboxes



def KmeanClustering(bboxes, max_iters, k = 5, plot = True) :
    '''
    ## https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
    read out whole bounding boxes from train datasets, and perform k mean clusterings
    bboxes = (x_center, y_center, w, h). Here ratios are respect to the image size, NOT grid cell.
    max_iters : number of maximum iterations
    k : number of clusters
    '''
    

    # initialize the centroid
    min_w, min_h = np.min(bboxes[:,2]), np.min(bboxes[:,3])
    max_w, max_h = np.max(bboxes[:,2]), np.max(bboxes[:,3])
    # uniformly distribute the k cluster centroids
    centroids = np.array([[np.random.uniform(min_w, max_w), np.random.uniform(min_h, max_h)] for i in range(k)])
    
    prev_centroids = None
    iteration = 0

    print('initial centroids : ', centroids)
    if plot :
        plt.scatter(bboxes[:,2], bboxes[:,3])
        plt.title('original points')
        plt.show()

    while np.not_equal(centroids, prev_centroids).any() or iteration < max_iters :
        
        sorted_points = [[] for _ in range(k)]

        for (_, _, w, h) in bboxes :
            
            # Suppose x_center and y_center are same for centroid and data point here, we just want the ratio of w, h.
            x, y = 0.5, 0.5

            dist = np.array([IoU(torch.tensor([x, y, w, h]), 
                                 torch.tensor([x, y, centroid_w, centroid_h ])) 
                                 for centroid_w, centroid_h in centroids]) # IoU supports only torch for now
            centroid_idx = np.argmax(dist)
            sorted_points[centroid_idx].append([w, h])

        prev_centroids = centroids
        centroids = [[np.mean([c[0] for c in cluster]), np.mean([c[1] for c in cluster])] for cluster in sorted_points ]

    if plot :
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(k)]

        for i, cluster in enumerate(sorted_points) :
            if len(cluster) :
                plt.scatter(np.array(cluster)[:,0], np.array(cluster)[:, 1], color = color[i])
        plt.title('after clustering')
        plt.show()

        # if any new centroids has NaN, then substitute back to previous centroids
        # centroids[np.isnan(centroids)] = prev_centroids[np.isnan(centroids)]

        iteration += 1

    return centroids

    # with open('anchorbox.txt', 'w') as f :
    #     f.write('\t'.join([str(anchorbox) for anchorbox in anchorboxes]))
        