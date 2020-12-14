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
def read_tensor_from_image_file(file_name,
                                input_height=224,
                                input_width=224,
                                input_mean=127.5,
                                input_std=127.5):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.io.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.io.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def add_jpeg_decoding(input_width=224, input_height=224, input_depth=3, input_mean=127.5,
                      input_std=127.5):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image

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
        for f in tqdm(test_img_list):
            im=Image.open(os.path.join(src,f))
            im = im.resize((224,224),Image.ANTIALIAS)
            #print(im.size)
            img=im.convert('RGB')
            img.save(os.path.join(dest,f))
            image_data = load_image(os.path.join(dest,f))

            img_data_jpg = tf.image.decode_jpeg(image_data) #图像解码
            img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
 
            # file_name = os.path.join(src,f)
            # t = read_tensor_from_image_file(
            # file_name)

            # decode_jpeg_data = tf.placeholder(dtype=tf.string)
            # decode_jpeg = tf.image.decode_jpeg(image_data)
            # image_data = sess.run(decode_jpeg,feed_dict={decode_jpeg_data: image_data})
            softmax_tensor=sess.graph.get_tensor_by_name(output_layer_name)
            predictions,=sess.run(softmax_tensor, {input_layer_name: tf.expand_dims(img_data_jpg, 0).eval()})
            #predictions,=sess.run(softmax_tensor, {input_layer_name: t})
            # print(predictions)
            # # Sort to show labels in order of confidence             
            # top_k = predictions.argsort()[-num_top_predictions:][::-1]
            # result=[]
            # for node_id in top_k:
            #     score = predictions[node_id]
            #     result.append(int(labels[node_id]))
            # csv_writer.writerow([f,result])
            csv_writer.writerow(predictions)
            i+=1



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

src="/home/DATA/guoming/food_recog/raw_data/test"
dest="/home/DATA/guoming/food_recog/raw_data/test2"
labels='./output/output_labels.txt'
predictions_csv="./output/mobilenet1.0_224_12_array.csv"
graph='./output/model_dir/output_graph.pb'
input_layer='DecodeJpeg/contents:0'
input_layer='input:0'
output_layer='final_result:0'
num_top_predictions=3
labels = load_labels(labels)
print("labels")
print(labels)
load_graph(graph)
# print("*************")
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#     print(tensor_name, '\n')
# print("__________")
run_graph(src,dest,labels,input_layer,output_layer,num_top_predictions,test_list,predictions_csv)
