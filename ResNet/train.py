import os
import cv2
import csv
import argparse
import numpy as np
import pandas as pd
import keras

from collections import Counter

from keras.callbacks import Callback
from keras.backend import clear_session
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten
from keras.applications import ResNet50

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from keras_model import build_model
num_classes = 1000


def F1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples')

class ComputeF1(Callback):
    
    def __init__(self):
        self.best_f1 = -1
        
    def on_epoch_end(self, epoch, logs={}):
        val_pred = np.round(self.model.predict(self.validation_data[0]))
        val_f1 = f1_score(self.validation_data[1], val_pred, average='samples')
        print('Validation Average F1 Score: ', val_f1)
        
        if val_f1 > self.best_f1:
            print('F1 Score is better, Saving model...')
            self.model.save('model.h5')
            self.best_f1 = val_f1


def load_data(str):

    trainX, trainY = [], []
    path = os.path.join('course_data/MTFood-1000', str)
    train_set = [os.path.join(path, file) for file in os.listdir(path)]

    for file in train_set:
        ri=file.rindex('/')
        item=file[ri+1:]
        current_label = item[:item.rfind('_')]
        trainY.append(current_label)
        img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
        trainX.append(img)
    return (np.array(trainX), np.array(trainY))

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices = ['ResNet50'],
                        help = 'model', required = True)

    args = parser.parse_args()

    print('Loading Data...')

    with open('./course_data/MTFood-1000/train_list','r') as f:
        train = f.read().splitlines()

    
    with open('./course_data/MTFood-1000/val_list','r') as t:
        val = t.read().splitlines()
    


    trainX, trainY = load_data("train")
    print("train loaded")

    valX, valY = load_data("val")
    print('Data Loaded.')


    trainX = trainX.astype(np.float32)
    valX = valX.astype(np.float32)

    trainY = trainY.astype(np.float32)
    valY = valY.astype(np.float32)

    MEAN = np.mean(trainX, axis = (0,1,2))
    STD = np.std(trainX, axis = (0,1,2))
    for i in range(3):
        print(MEAN[i])
    print("STD:")
    for j in range(3):
        print(STD[i])
    '''
    trainY = tf.one_hot(trainY, num_classes)
    valY = tf.one_hot(valY, num_classes)
    '''
    trainY = keras.utils.to_categorical(trainY, num_classes)
    valY = keras.utils.to_categorical(valY, num_classes)

    for i in range(3):
        trainX[:, :, :, i] = (trainX[:, :, :, i] - MEAN[i]) / STD[i]
        valX[:, :, :, i] = (valX[:, :, :, i] - MEAN[i]) / STD[i]

    f1_score_callback = ComputeF1()
    
    model = build_model('train', model_name = args.model)

    
    ## Training model.
    model.fit(trainX, trainY, batch_size = 32,  epochs = 25, validation_data = (valX, valY), 
              callbacks = [f1_score_callback])
    model = load_model('model.h5')

    '''
    score = F1_score(testY, model.predict(testX).round())
    print('F1 Score =', score)
    '''

