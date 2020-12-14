from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import argparse
import sys,os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import csv

t=tqdm(pd.read_csv('/home/DATA/guoming/food_recog/raw_data/list/food_meituan_test.csv').values)
test_list=[]
i=0
for tt in t:
    test_list.append(tt[0])
    i+=1

def load_image(filename):
    #Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
    #Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_graph(filename):
    #Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')



def run_graph(src, dest, labels, input_layer_name, output_layer_name, num_top_predictions,test_img_list,output_csv):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        # Feed the image_data as input to the graph.
        # predictions  will contain a two-dimensional array, where one
        # dimension represents the input image count, and the other has
        # predictions per class
        i=0
        f_csv = open(output_csv,'w',encoding='utf-8',newline="")
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["id","predicted"])
        #with open('submit.csv','w') as outfile:
        for f in test_img_list:
            im=Image.open(os.path.join(src,f))
            img=im.convert('RGB')
            img.save(os.path.join(dest,f))
            image_data=load_image(os.path.join(dest,f))
            softmax_tensor=sess.graph.get_tensor_by_name(output_layer_name)
            predictions,=sess.run(softmax_tensor, {input_layer_name: image_data})
            
            # Sort to show labels in order of confidence             
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            result=[]
            for node_id in top_k:
                score = predictions[node_id]
                result.append(labels[node_id])
            csv_writer.writerow([f,result])
            i+=1


src="/home/DATA/guoming/food_recog/raw_data/test"
dest="/home/DATA/guoming/food_recog/raw_data/test2"
labels='./output/output_labels.txt'
predictions_csv="./output/guoming_test.csv"
graph='./output/model_dir/output_graph.pb'
input_layer='DecodeJpeg/contents:0'
output_layer='final_result:0'
num_top_predictions=3
labels = load_labels(labels)
print("labels")
print(labels)
load_graph(graph)
run_graph(src,dest,labels,input_layer,output_layer,num_top_predictions,test_list,predictions_csv)
