import os
import cv2
import argparse
import numpy as np
import pandas as pd
import csv

from collections import Counter

from keras.callbacks import Callback
from keras.backend import clear_session
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten
from keras.applications import ResNet50

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from keras_model import build_model


def load_testdata():
    testX, testname = [], []
    path = './course_data/MTFood-1000/test'
    test_set = [os.path.join(path, file) for file in os.listdir(path)]

    for file in test_set:
        img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
        testX.append(img)
        testname.append(file)
        
    return (np.array(testX), np.array(testname))

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    MEAN = np.array([1.1368206, 1.1368206, 1.1368206])
    STD = np.array([17.059486,  17.059486,  17.059486])
    categories = []
    for i in range(1000):
        categories.append(i)

    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', help = 'Path of the saved model', required = True)

    args = parser.parse_args()

    model = build_model('inference', model_path = args.saved_model)

    testX, testname = load_testdata()
    print("test loaded")
    testX = testX.astype(np.float32)
    for i in range(3):
        testX[:, :, :, i] = (testX[:, :, :, i] - MEAN[i]) / STD[i]
   
    f = open('test.csv','w')
    csv_writer = csv.writer(f)

    csv_writer.writerow(["id","predicted"])

    res = model.predict(testX).round()
    
    for i in range(len(res)):
        labels = [categories[idx] for idx, current_prediction in enumerate(res[i]) if current_prediction == 1]
        csv_writer.writerow([testname[i],labels[0:2]])

