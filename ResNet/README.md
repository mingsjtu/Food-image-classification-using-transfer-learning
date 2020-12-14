# Meituan Food recognization: ResNet50 by QIAN Chen

## 1. Prepare dataset

The structure of your dataset should be:

- ./course_data/MTFood-1000/train
- ./course_data/MTFood-1000/test
- ./course_data/MTFood-1000/val

## 2. Train model

- Our pretrain model saved as model.h5

- Train your model from scratch (and maybe more epochs):

  run

  ```
  python train.py --model ResNet50
  ```

- In each epoch, the F1 score will be computed and compared. If the new result is better, the model will be saved as model.h5.

## 3. Test model

- Make sure your dataset follow the path required above.

- run

  ```
  python predict.py --saved_model PATH_TO_model
  ```

  If you use ResNet50 model just trained above, you can change "PATH_TO_model" to model.h5. Or you can just use predict.py to predict your model result.

- As required, the result will be showed in test.csv. It gives the top3 based on probabilities.

## Reference

[1]He K , Zhang X , Ren S , et al. Deep Residual Learning for Image Recognition[C]// 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2016.