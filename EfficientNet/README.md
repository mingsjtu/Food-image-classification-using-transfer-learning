# Meituan Food recognization: EfficientNet model by Guo Ming

## 1. Prepare dataset
The structure of your dataset should be:
- dataset root
    - train\
         - class1\
         - class2\
         ...
    - val\
         - class1\
         - class2\
         ...
    - test\
         - 0\
- No meaning for test class name

## 2. Train model

- Download pretrained model
The pre-trained model is available on >[release](https://github.com/lukemelas/EfficientNet-PyTorch/releases). 

You can choose b1,b2,...,b7

And store pre-trained model in pre_trained dir

- modify the data path in [train.py](train.py)
- run
  ```
  python train.py
  ```
- You can get the best model on ```dataset/model/```
## 3. Test model

- modify the data path and model path in [test.py](test.py)
- run
  ```
  python test.py
  ```
- The label and the index correspondence can be found in [label_np.npy](label_np.npy)

## 4. Our accuracy
The best top3 test accuracy of this model is 85.0%

## Reference
[1]	Tan M, Le QV. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946. 2019 May 28.