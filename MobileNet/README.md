# Meituan Food recognization: Google mobile net by Fu Yu

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
The pre-trained model is available on >[release](http://download.tensorflow.org/models/mobilenet_v1_). 

It will be download to ```pre_model_dir``` automatically

- modify the data path in [train.py](train.py)
- run
  ```
  python train.py
  ```
- You can the best model on ```output/model_dir```

## 3. Validate model
- modify the data path and model path in [val.py](val.py)
- run
  ```
  python val.py
  ```
- The result will be saved in a csv file
## 4. Test model

- modify the data path and model path in [test.py](test.py)
- run
  ```
  python test.py
  ```
- The result will be saved in a csv file

## 4. Our accuracy
The best top3 test accuracy of this model is 59.1%

## Reference
[1]	Howard AG, Zhu M, Chen B, Kalenichenko D, Wang W, Weyand T, Andreetto M, Adam H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861. 2017 Apr 17.

