## YOLO9000: Better, Faster, Stronger
[blog post](https://silvercity.oopy.io/61538bf0-b66a-4ca4-be3d-a9b80e22c4f5)  

### new features  
- batch normalization
- higher resolution backbone (DarkNet19)
- convolutional layer at the end, instead of Fully Connected Layer
- anchor boxes (from k-means clustering; 1-IoU)
- direct location prediction
- 13x13 grid
- passthrough layer like ResNet at the end
