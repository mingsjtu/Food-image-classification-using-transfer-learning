# Meituan Food recognization: 

# Data Preprocessing by QIAN Chen

## 1. Prepare dataset

Modify the variance ***IMAGE_PATH*** to your dataset path.

## 2. Process the data

- If you want to **rescale** and **random crop** the images in the dataset, you can just run process.py as:

  ```
  python process.py
  ```

- If you want to accomplish data argumentation, you can also use **horizontal_flip** function.

## Notification

You should notice that the image will save to the original path. Please backup your data before preprocessing.