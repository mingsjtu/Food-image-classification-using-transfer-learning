# Food Recognize Project by G21


1. This is the project for course "大数据的商业应用与实践" by Tsinghua University and MeiTuan.

2. The kaggle link can be found [here](https://www.kaggle.com/c/2020-cv-veg)

3. The dataset we call MTFood-1000 is provided by Meituan (thanks a lot).
   The basic information of the dataset:
   - class number:1000
   - train set: 75297
   - val set: 10000
   - test set: 20000

4. The model we use are:
   - ResNet
   - Inception-v3
   - MobileNet
   - EfficientNet

## The structure of the project
1. Data preprocessing[Data_preprocessing](Data_preprocessing/README.md)
2. ResNet[ResNet](ResNet/README.md)
3. Inception-v3 [Inception-v3](Inception-v3/README.md)
4. MobileNet [MobileNet](MobileNet/README.md)
5. EfficientNet [EfficientNet](EfficientNet/README.md)
   

For more information of each model, we provide a detailed README in each part for you to reference.


## The Result of Our project

| Model      | Train top1 accuracy    | Test top3 accuracy  |
| --------   | -----:   | :----: |
| ResNet-50        | --      |   25.6%    |
| Inception-v3      | 65.3%      |   61.1%    |
| MobileNet        | 78.5%      |   59.1%    |
| EfficientNet        | 90.1%      |   77.2%    |

## Thanks to
Thanks to teachers from Tsinghua University and Meituan


## Reference
[1]	He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).

[2]	Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the inception architecture for computer vision. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 2818-2826).

[3]	Howard AG, Zhu M, Chen B, Kalenichenko D, Wang W, Weyand T, Andreetto M, Adam H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861. 2017 Apr 17.

[4]	Tan M, Le QV. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946. 2019 May 28.
