# 연구노트 7주차 (8/14 - 8/18)
## 활동 내용
Continual Learning
 

## 논문 Review
| Week   | Paper                                               | Conf | Year   | Review   |
| :----: | ------------------------------------------------------- | :----: | :------------: | :------: |
| 7   |  [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)<br>[Methods for interpreting and understanding deep neural networks](https://www.sciencedirect.com/science/article/pii/S1051200417302385)<br>[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)   | ICLR<br>DSP<br>ICCV    | 2021<br>2018<br>2017   | [Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week7/Reviews/AN%20IMAGE%20IS%20WORTH%2016X16%20WORDS%20TRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE.pdf)<br>[Review]()<br>[Review]() |

### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
Image Recognition에 Transformer를 이용하기 위한 시도는 이전에도 있었지만, 이 논문에서는 image-specific bias를 이용하기 보다는 이미지를 sequence of patch로서 만들어 학습시켰다. 이미지를 부분으로 쪼갠 덕분에 Transformer에서 부족한 Inductive Bias를 활용할 수 있었으며, Position Embedding을 활용하여 위치 정보도 함께 학습시켰다. 그 결과 기존 NLP 학습방법과 거의 유사하게 학습시켰음에도 불구하고 좋은 Image Recognition에서 좋은 결과를 얻을 수 있었다.

### Methods for interpreting and understanding deep neural networks

### Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization


## 멘토멘티 프로젝트
[Challenge](https://www.kaggle.com/competitions/cilab-summer-intern-program-challenge/)  
[Assignment 6](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week7/Project/week6.pdf)

## 개인 프로젝트

