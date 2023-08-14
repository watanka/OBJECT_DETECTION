## YOLOv3: An Incremental Improvement
[blog post](https://silvercity.oopy.io/2115737e-1872-4267-a1bd-7d5b91c3db23)
### model architecture
![img](../img/yolov3_architecture.png) 


#### new features
- class prediction, use binary cross-entropy. multi-label
- predicts boxes at 3 different scales, 3boxes at each scale
    - feature map from 2 layers previous and upsample it by 2x
- DarkNet-53
