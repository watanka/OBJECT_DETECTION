import torch



## convert labelgrid into (bboxes, confidence_scores, labels)
def convert_prediction(label_grid, num_bboxes, num_classes) :
    '''
    operations are grid units.
    
    MAKE SURE label_grid is NOT BATCHWISE.

    label_grid : output of the model. (num_grid x num_grid x (num_bboxes * 5 + num_classes ) 
    num_bboxes : maximum number of boxes per grid cell
    num_classes : # of classes
    '''

    num_grid, num_grid, seq_length = label_grid.shape

    # get class index with highest probabilities
    classlist_grid = label_grid[..., num_bboxes*5 : ]
    max_probability_class_grid = torch.argmax(classlist_grid, -1)

    # select coordinates and confidence scores with highest probability per grid cell
    coords_grid = label_grid[..., :num_bboxes*5]
    
    confidence_score_idx = [5*(i+1)-1 for i in range(num_bboxes)] # select only the confidence score indexes
    max_confidence_grid, idx_grid = torch.max(coords_grid[..., confidence_score_idx], keepdim= True, dim = -1)

    ## idx_grid refers to the index of box information with the highest confidence score. since [x,y,w,h, conf_score]. we divide confidence score divided by 5.
    coords_idx_grid = torch.cat([(idx_grid // 5) + i for i in range(5)], -1) 

    xywh_grid = torch.gather(coords_grid, -1, coords_idx_grid)

    ## convert ratio respect to the image size => add y0, x0 coordinates. (y0, x0) differ by grid cell.
    gridsize = 1 / num_grid
    bboxes = []
    for j in range(num_grid) :
        for i in range(num_grid) :
            y0 = gridsize * j
            x0 = gridsize * i
            x, y, w, h, confidence_score = xywh_grid[j][i].detach().cpu().numpy()
            pred_cls = max_probability_class_grid[j][i].detach().cpu().numpy()

            bboxes.append([
                            pred_cls, x + x0, y + y0, w, h, confidence_score
                            ])

    

    return bboxes





