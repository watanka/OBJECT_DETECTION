import torch


## convert labelgrid into (confidence_scores, bboxes, labels)
def convert_labelgrid(label_grid, numbox, num_classes):
    """
    operations are grid units.

    MAKE SURE label_grid is NOT BATCHWISE.

    label_grid : output of the model. (num_grid x num_grid x (numbox * 5 + num_classes )
    numbox : maximum number of boxes per grid cell
    num_classes : # of classes
    """

    num_grid, num_grid, seq_length = label_grid.shape

    # get class index with highest probabilities
    classlist_grid = label_grid[..., numbox * 5 :]
    max_probability_class_grid = torch.argmax(classlist_grid, -1)

    # select coordinates and confidence scores with highest probability per grid cell
    coords_grid = label_grid[..., : numbox * 5]

    # select only the confidence score indexes
    confidence_score_idx = [i for i in range(numbox)]  
    max_confidence_grid, idx_grid = torch.max( coords_grid[..., confidence_score_idx], keepdim=True, dim=-1)

    ## idx_grid refers to the index of box information with the highest confidence score. since [conf_score,x,y,w,h]. we divide confidence score divided by 5.
    coords_idx_grid = torch.cat([idx_grid + i for i in range(5)], -1)

    cxywh_grid = torch.gather(coords_grid, -1, coords_idx_grid)

    ## convert ratio respect to the image size => add y0, x0 coordinates. (y0, x0) differ by grid cell.
    gridsize = 1 / num_grid
    bboxes = []
    for j in range(num_grid):
        for i in range(num_grid):
            y0 = gridsize * j
            x0 = gridsize * i
            confidence_score, x, y, w, h  = (
                cxywh_grid[j][i].detach().cpu().numpy().tolist()
            )
            pred_cls = max_probability_class_grid[j][i].detach().cpu().numpy().tolist()

            bboxes.append([confidence_score, x + x0, y + y0, w, h, pred_cls])

    return bboxes

def decode_labelgrid(label_grid, numbox, num_classes) :
    '''For GT label conversion, separate from prediction conversion since gt label grid can have numBox true bboxes'''
    num_grid, num_grid, seq_length = label_grid.shape

    # get class index with highest probabilities
    classlist_grid = label_grid[..., numbox * 5 :]
    max_probability_class_grid = torch.argmax(classlist_grid, -1)

    # select coordinates and confidence scores with highest probability per grid cell
    coords_grid = label_grid[..., : numbox * 5]

    ## convert ratio respect to the image size => add y0, x0 coordinates. (y0, x0) differ by grid cell.
    gridsize = 1 / num_grid
    bboxes = []
    for j in range(num_grid):
        for i in range(num_grid):
            y0 = gridsize * j
            x0 = gridsize * i
            for b in range(numbox) :
                confidence_score, x, y, w, h  = (
                    coords_grid[j][i][b*5 : (b+1)*5].detach().cpu().numpy().tolist() # get each bbox from same grid
                )
                pred_cls = max_probability_class_grid[j][i].detach().cpu().numpy().tolist()

                bboxes.append([confidence_score, x + x0, y + y0, w, h, pred_cls])

    return bboxes