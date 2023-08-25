# 연구노트 5주차 (7/31 - 8/4)
## 활동 내용
Detection & Segmentation 논문 Review  
 

## 논문 Review
| Week   | Paper                                               | Conf | Year   | Review   |
| :----: | ------------------------------------------------------- | :----: | :------------: | :------: |
| 5 & 6   | [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)<br>[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)<br>[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)<br>[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)<br>[Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)       | CVPR<br>NIPs<br>CVPR<br>ECCV<br>CVPR  | 2014<br>2015<br>2016<br>2016<br>2017    | [Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week5/Reviews/Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation.pdf)<br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week5/Reviews/Faster%20R-CNN%20Towards%20Real-Time%20Object%20Detection%20with%20Region%20Proposal%20Networks.pdf)<br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week5/Reviews/You%20Only%20Look%20Once%20Unified%2C%20Real-Time%20Object%20Detection.pdf)<br><br><br> |

### Rich feature hierarchies for accurate object detection and semantic segmentation
Object Detection에 CNN을 사용하면 속도와 성능을 쉽게 개선할 수 있음을 밝혀내고 이를 활용한 모델인 R-CNN을 제안한 논문이다. R-CNN은 Class와 관계없는 Region들을 만든 뒤, 그 Region들을 CNN에 넣어서 분류하는 방식을 사용하였다. 이 논문에서는 또한 당시 라벨된 데이터 부족 현상을 해결하기 위해 fine-tuning을 활용하였다.

### Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
기존의 Fast R-CNN에 RPN을 도입하여 Selective Search를 대체하고 학습이 가능한 방식으로 더 빠르고 정확한 ROI를 찾아낼 수 있도록 한 논문이다. 이러한 RPN은 여러가지 크기의 Anchor를 Sliding Window 기법으로 Conv Feature Map 위를 한 pixel씩 옮겨가면서 찾게 된다. Fast R-CNN과 RPN이 같은 Conv Net을 공유하기 때문에, 이를 학습시키기 위해서는 특수한 4-Step Alternating Training이라는 기법을 사용한다. 그 결과 Faster R-CNN은 5에서 17 fps의 속도로 Object Detection을 수행할 수 있게 되었다. [Presentation](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week5/Reviews/Faster%20R-CNN_Towards%20Real-Time%20Object%20Detection%20with%20Region%20Proposal%20Networks.pptx)  

### You Only Look Once: Unified, Real-Time Object Detection
Object Detection을 실시간으로 진행할 수 있는 YOLO를 제안한 논문이다. YOLO는 Unified Detection 방식으로 Object Detection을 진행하며, Multi-Task 문제를 Regression 문제로 재정의한 뒤 Grid Cell과 Bounding Box를 활용한다. 그 결과 45 FPS라는 빠른 처리 속도를 확보할 수 있었다. 그러나 YOLO는 Localization 문제가 발생하였고 작은 물체를 잘 탐지하지 못하는 문제가 있어서 실질적인 성능은 조금 떨어질 수 있다.

## 멘토멘티 프로젝트

[Assignment 4](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week5/Project/week4.pdf)  
[보고서](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week5/Project/Assignment4.pdf)  
[WandB](https://wandb.ai/oso0310/project4/reports/Assignment-4--Vmlldzo1MDY3NzU0)  
[코드](https://github.com/Chihiro0623/2023summer-selfstudy1/tree/main/week5/Project/Assignment4)  

[Challenge](https://www.kaggle.com/competitions/cilab-summer-intern-program-challenge/)  

## 산학협력 프로젝트
[Poisson Image Augmentation](https://github.com/Chihiro0623/Defect-Prediction-by-CNN/tree/main/poisson-image-editing-master)
