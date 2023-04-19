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
        - channel-wise attention : Squeeze-and-Excitation(SE) => increases only 2% computational effort and improve 1% top-1 accuracy, but result in 10% increase in the inference time in GPU (attention not parallelized)
        - point-wise attention : Spatial Attention Module (SAM) : 0.1% extra calculation and improve 0.5% top-1 accuracy. No influence on GPU inference speed.
    - Feature Integration
        - skip connection
        - hyper-column
        - FPN and more lightweight feature pyramid integration module like SFAM, ASFF, BiFPN
            - SFAM : use SE module to execute channel-wise level re-weighting on multi-scale concatenated features maps
            - ASFF : uses softmax as point-wise level reweighting and then adds feature maps of different scales
            - BiFPN : multi-input weighted residual connections is proposed to execute scale-wise level re-weighting, and then add feature maps of different scales
    - Activation Function
        - ReLU
        - LReLU - to solve the problem that ReLu cannot cover the value less than zero
        - PReLU - to solve the problem that ReLu cannot cover the value less than zero
        - ReLU6 - specially designed for quantization network
        - SeLU(Scaled Exponential Linear Unit) - for self-normalizing neural network
        - Swish (continuously differentiable activation function)
        - hard Swish - specially designed for quantization network
        - Mish (continuously differentiable activation function)
    - Network
        - CSPDarknet53
        - CSPResNext50
        - EfficientNet-B3

    - post-processing method
        - NMS
        - soft NMS : considers the problem that the occlusion of an object may cause the degradation of confidence score in greedy NMS with IoU scores
        - DIoU NMS : add the information of the center point distance to the bbox screening process on the basis of soft NMS



### Selection

Head  
- CSPDarkNet53
Neck  
- SPP, PAN  
Head  
- YOLOv3



