## YOLO9000: Better, Faster, Stronger
[blog post](https://silvercity.oopy.io/acfeb321-c9f6-4722-a7d6-2ba3acb36f8f)  

### new features  
- batch normalization
- higher resolution backbone (DarkNet19)
- convolutional layer at the end, instead of Fully Connected Layer
- anchor boxes (from k-means clustering; 1-IoU)
- direct location prediction
- 13x13 grid
- passthrough layer like ResNet at the end