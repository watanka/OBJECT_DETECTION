## YOLO9000: Better, Faster, Stronger
[blog post](https://silvercity.notion.site/YOLO9000-Better-Faster-Stronger-61538bf0b66a4ca4be3da9b80e22c4f5)  

### new features  
- batch normalization
- higher resolution backbone (DarkNet19)
- convolutional layer at the end, instead of Fully Connected Layer
- anchor boxes (from k-means clustering; 1-IoU)
- direct location prediction
- 13x13 grid
- passthrough layer like ResNet at the end
