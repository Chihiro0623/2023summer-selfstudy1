# 연구노트 7주차 (8/14 - 8/18)
## 활동 내용
Continual Learning
 

## 논문 Review
| Week   | Paper                                               | Conf | Year   | Review   |
| :----: | ------------------------------------------------------- | :----: | :------------: | :------: |
| 7   |  [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)<br>[Methods for interpreting and understanding deep neural networks](https://www.sciencedirect.com/science/article/pii/S1051200417302385)<br>[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)   | ICLR<br>DSP<br>ICCV    | 2021<br>2018<br>2017   | [Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week7/Reviews/AN%20IMAGE%20IS%20WORTH%2016X16%20WORDS%20TRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE.pdf)<br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week7/Reviews/Methods%20for%20interpreting%20and%20understanding%20deep%20neural%20networks.pdf)<br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week7/Reviews/Grad-CAM%20Visual%20Explanations%20from%20Deep%20Networks%20via%20Gradient-based%20Localization.pdf) |

### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
Image Recognition에 Transformer를 이용하기 위한 시도는 이전에도 있었지만, 이 논문에서는 image-specific bias를 이용하기 보다는 이미지를 sequence of patch로서 만들어 학습시켰다. 이미지를 부분으로 쪼갠 덕분에 Transformer에서 부족한 Inductive Bias를 활용할 수 있었으며, Position Embedding을 활용하여 위치 정보도 함께 학습시켰다. 그 결과 기존 NLP 학습방법과 거의 유사하게 학습시켰음에도 불구하고 좋은 Image Recognition에서 좋은 결과를 얻을 수 있었다.

### Methods for interpreting and understanding deep neural networks
이 논문은 심층 신경망 모델을 해석 및 이해하고 예측을 설명하는 방법들을 소개한 tutorial이다. 주로 LRP를 중심으로 DNN을 해석하였는데, LRP란 forward pass에서 각 레이어의 활성화 값을 기록한 뒤 backward pass에서 레이어별로 점수를 분배하여 각 입력 변수가 출력 값에 얼마나 기여하는지를 추적하고 시각화하는 방식이다. 모델이 어떻게 작동하는지 이해하는 것은 모델의 신뢰성을 올리기 때문에 자율주행 자동차 등에 적용하기에 앞서 필수적으로 진행되어야 한다.

### Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
심층 신경망을 시각화할 수 있는 Grad-CAM을 제시한 논문이다. Grad-CAM은 CNN의 마지막 Conv Layer에 흘러 들어오는 gradient를 활용하여 이미지화한다. Grad-CAM은 CNN의 아키텍처를 변경하거나 재교육시킬 필요 없이 바로 시각화할 수 있다는 장점이 있다. Image classification, image captioning, visual questioning answering과 같은 다양한 task들에서 attention이 없는 모델에서도 잘 작동함을 보여주었다.

## 멘토멘티 프로젝트
[Challenge](https://www.kaggle.com/competitions/cilab-summer-intern-program-challenge/)  
[Assignment 6](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week7/Project/week6.pdf)  
[보고서](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week7/Project/Assignment6.pdf)  
[코드](https://github.com/Chihiro0623/2023summer-selfstudy1/tree/main/week7/Project/Assignment6)  

## 개인 프로젝트
[CaiT 논문 읽기](https://github.com/Chihiro0623/ContinualLearning/blob/main/Study/Papers/Going%20deeper%20with%20Image%20Transformers.pdf)  

