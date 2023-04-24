자세한 내용은 블로그에 정리해두었습니다.  
[blog](https://silvercity.notion.site)

#### reference
https://arxiv.org/pdf/2304.00501.pdf  


![img](img/yolo_timeline.png)  

### Description
yoloV1부터 yoloV4까지 직접 구현합니다. 이미 웹에 pytorch로 구현한 코드들이 많이 존재했기 때문에, 최대한 활용은 하였으나, 코드들은 전부 제 이해를 바탕으로 직접 작성한 코드들입니다. 
그럼에도 불구하고, 참고한 코드들이 없었다면, 사용가능한 코드까지 끝마치지는 못 했을 거라는 생각이 듭니다.  

### Motivation
논문을 읽고 이해했다고 '생각'했던 것들을 직접 구현해보면서, 지식의 빈 부분들을 채울 수 있었습니다. 또한, 버젼이 업그레이드됨에 따라 Object Detection 성능향상을 위해 학계에 어떤 시도들이 있었는지 흐름을 파악하고 구현할 줄 안다면, 다른 object detection 모델을 개발해야할 때, 어떤 부족한 점이 있고, 개선하기 위해서 어떤 시도들을 해야할지 알 수 있겠다는 생각이 들었습니다.   

[o] [yolov1](YOLOv1/README.md)
[o] [yolov2](YOLOv2/README.md)  
[o] [yolov3](YOLOv3/README.md)  
[o] [yolov4](YOLOv4/README.md)



#### Requirements
- pytorch
- pytorch_lightning
- albumentations
- hydra  

#### Dataset
Berkeley DeepDrive Dataset  
- link : https://bdd-data.berkeley.edu/  
- pascal_voc 형태, [x1,y1,x2,y2]를 인풋으로 받습니다. albumentations에서 yolo format을 처리할 시에 생기는 에러 때문.  
- [objectness, x, y, w, h, classes].


#### Issue
04-14  
- pretrained darknet을 찾지 못함. detection 모델 학습 전, classification task를 사용해서 backbone network인 darknet을 pretrain시킴. RTX2070 하나로 큰 pretrained model을 직접 학습시키기에는 무리가 있음. 우선 pretrained 없이 바로 detection 모델 학습을 진행하기로 함.
04-14  
- 결괏값이 (confidence score, x, y, w, h, class), ...  이런 형태로 되어있어 가독성이 떨어지고 디버깅하기 어려움. dictionary로 각 박스에 따라 섹션을 나눠 분배하면 가독성이 조금 개선될듯함.  