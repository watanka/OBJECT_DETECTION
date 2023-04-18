## YOLOv4: Optimal Speed and Accuracy of Object Detection

#### new features
- Bag of Freebies
    - Data Augmentation
        - photometric distortion; brightness, contrast, hue, saturation, noise
        - geometric distortion; random scaling, cropping, flipping, and rotating
        - random erase
        - CutOut
        - hide-and-seek and grid mask
        - DropOut, DropConnect, DropBlock
        - MixUp, CutMix
        - styleGAN
    - For Semantic Distribution in the data
        - focal loss
        - label smoothing
    - Bounding Box Regression
        - IoU loss
            - GIoU loss
            - DIoU loss
            - CIoU loss
- Bag of Specials
    - enhance receptive field
        - SPP(Spatial Pyramid Pooling; implemented in yolov3)
        - ASPP(Atrous Spatial Pyramid Pooling)
        - RFB(Receptive Field Block)
    - Attention Module
        - 