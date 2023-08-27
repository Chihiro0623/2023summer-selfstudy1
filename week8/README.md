# 연구노트 8주차 (8/21 - 8/25)
## 활동 내용
Challenge  
Continual Learning 공부
 

## 논문 Review
| Week   | Paper                                               | Conf | Year   | Review   |
| :----: | ------------------------------------------------------- | :----: | :------------: | :------: |
|8    | [Intriguing Properties of Vision Transformers](https://arxiv.org/pdf/2105.10497.pdf)<br>[ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/pdf/1811.12231.pdf)<br>[How Do Vision Transformers Work?](https://arxiv.org/pdf/2202.06709.pdf)   |   NIPs<br>ICLR<br><br>ICLR  | 2021<br>2019<br><br>2022 | [Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week8/Reviews/Intriguing%20Properties%20of%20Vision%20Transformers.pdf)<br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week8/Reviews/ImageNet-trained%20CNNs%20are%20biased%20towards%20texture%3B%20increasing%20shape%20bias%20improves%20accuracy%20and%20robustness.pdf)<br><br>[Review](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week8/Reviews/How%20Do%20Vision%20Transformers%20Work.pdf ) |

### Intriguing Properties of Vision Transformers
Image 처리에 사용된 Transformer의 다양한 속성들, 특히 robustness와 generalizability를 중심으로 파악하고 이를 CNN과 비교한 논문이다. ViT는 이미지를 80%를 가리는 상황에서도 60%의 성능을 내는 등의 높은 robustness를 보여주며, 이는 CNN과 대조적이다. 또한 ViT는 Texture보다 Shape를 중심으로 보며, 이 또한 CNN과 대조적이다. 특히 인상깊었던 내용은 사진을 여러 patch로 나누어 섞은 뒤 성능을 측정해보았을 때인데 당연히 Bias가 높은 CNN보다 Transformer가 더 높은 성능을 보여주었으며, 오히려 CNN을 따라하기 위해 넣은 positional encoding을 뺐을 때 더 성능이 개선되었다는 점이었다.

### ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness
ImageNet으로 학습된 CNN은 texture만을 보도록 편향될 수 있다는 것을 밝혀낸 논문이다. 심리 실험을 통해 사람은 shape를 중점적으로 보는데 반해 CNN은 texture를 보는 것을 알아냈다. 그러나 이러한 texture bais는 robustness에 제약을 받기 때문에 왜곡된 이미지를 판단하는데 약해질 수 있다. 따라서 Shape에도 집중할 수 있도록 학습시킬 수 있게 ImageNet의 모든 사진들의 Texture를 변형시킨 Stylized-ImageNet (SIN)이라는 Dataset을 만들었고, ImageNet으로 학습시킨 뒤 SIN과 ImageNet을 함께 이용해 Fine-Tuning할 경우 가장 높은 성능을 보임을 확인하였다.

### How Do Vision Transformers Work?
Vision Transformer와 Multi-head self-attention이 어떻게 작동하는지에 대한 기존의 통념을 반박하면서 새로운 아키텍처인 AlterNet을 제안한 논문이다. 기존에는 MSA가 성능을 향상시켰던 이유가 weak inductive bias와 long-range dependency 때문이라고 생각했으며, 따라서 small data에 대해서는 overfit이 발생하여 성능이 떨어진다고 생각하였으나, 적절한 inductive bias가 있는 것이 좋으며 MSA는 loss landscape를 flatten하고 convexify하고 data specificity를 올리기 때문에 성능을 개선한 것이고 small data에 대해서는 non-convexity 때문이지 overfit 때문이 아니라는 것을 밝혀냈다. 그리고 MSA와 Conv는 상반되게 작동하기 때문에 상호보완적으로 사용할 수 있으므로 이 둘의 장점만을 결합시킨 AlterNet을 제안하였다. [Presentation](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week8/Reviews/How%20Do%20Vision%20Transformers%20Work_.pptx)


## 멘토멘티 프로젝트
[Challenge](https://www.kaggle.com/competitions/cilab-summer-intern-program-challenge/)  
[Assignment 7](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week8/Project/week7.pdf)  
[보고서](https://github.com/Chihiro0623/2023summer-selfstudy1/blob/main/week8/Project/Assignment7.pdf)  
[코드](https://github.com/Chihiro0623/2023summer-selfstudy1/tree/main/week8/Project/Assignment7)  

