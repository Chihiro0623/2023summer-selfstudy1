# 연구노트 6주차 (8/7 - 8/11)
## 활동 내용
Challenge  
Continual Learning 공부
 

## 논문 Review
| Week   | Paper                                               | Conf | Year   | Review   |
| :----: | ------------------------------------------------------- | :----: | :------------: | :------: |
| 5 & 6   | [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)<br>[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)<br>[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)<br>[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)<br>[Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)       | CVPR<br>NIPs<br>CVPR<br>ECCV<br>CVPR  | 2014<br>2015<br>2016<br>2016<br>2017    | <br><br><br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week6/Reviews/SSD%20Single%20Shot%20MultiBox%20Detector.pdf)<br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week6/Reviews/Feature%20Pyramid%20Networks%20for%20Object%20Detection.pdf) |

### SSD: Single Shot MultiBox Detector
이미지에서 여러 카테고리를 효율적으로 감지하는 SSD를 제안한 논문이다. SSD는 미리 준비된 크기의 box를 통해 bounding box를 예측함으로서 proposal의 생성을 줄이고 resampling을 없애 속도를 더 빠르게 만들었고 학습을 쉽게 만들었다. 또한 여러 해상도의 feature maps로부터의 결과를 총합하기 때문에 더 작은 이미지 입력에 대해서도 잘 작동한다.

### Feature Pyramid Networks for Object Detection
Object Detection을 pyramid식 filter에서 벗어나 효율적인 예측을 위한 FPN을 제안한 논문이다. FPN은 bottom-up과 top-down의 두 개의 Pathway를 통해 Feature Pyramid Network를 만든다. Lateral Conection은 두 pathway 사이를 연결하여 top-down pathway에서 unsample된 feature map이 bottom-up pathway에서 추출된 semantic information을 잘 포함할 수 있도록 도와준다. 이런 방식을 활용하면 큰 개체는 pyramid의 낮은 레벨에서, 작은 개체는 높은 레벨에서 감지될 수 있다.

## 멘토멘티 프로젝트
[Challenge](https://www.kaggle.com/competitions/cilab-summer-intern-program-challenge/)  

## 개인 프로젝트
[iCarl 논문 읽기](https://github.com/Chihiro0623/ContinualLearning/blob/main/Study/Papers/iCaRL%20Incremental%20Classifier%20and%20Representation%20Learning.pdf)
