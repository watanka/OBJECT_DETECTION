## YOLOv3: An Incremental Improvement

#### new feature
- class prediction, use binary cross-entropy. multi-label
- predicts boxes at 3 different scales, 3boxes at each scale
    - feature map from 2 layers previous and upsample it by 2x
- DarkNet-53
