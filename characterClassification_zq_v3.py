# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:55:26 2018

@author: Administrator
"""
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import argparse
import sys
import os
import cv2 as cv
import time
import random
import numpy as np
from PIL import Image
from datetime import datetime
import Word_data as wd
import praseLabel2
import wordDetectLocation03 as worddect
SIZE = 1000  #32x40
WIDTH = 25
HEIGHT = 40
NUM_CLASSES = 68  #输入变量的个数
trainingTimes = 1000  #训练次数
batch_size = 2048
epochs = 20000
FLAGS = None
#Keep_prob = 0.9
 
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D",
                  "E","F","G","H","J","K","L","M","N","P","Q","R","S","T",
                  "U","V","W","X","Y","Z","京","闽","粤","苏","沪","浙","皖",
                  "渝","甘","桂","贵","琼","冀","黑","豫","鄂","湘","赣","吉","辽",
                  "蒙","宁","青","鲁","晋","陕","川","津","新","云","藏","港","澳", "wu")
license_num = ""



# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')

# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)

def characterClassificationNet(x_image):
    # 第一个卷积层
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="W_conv1")
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv1")
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 2, 2, 1]
    pool_strides = [1, 2, 2, 1]
    L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

    # 第二个卷积层
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="W_conv2")
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b_conv2")
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 2, 2, 1]
    pool_strides = [1, 2, 2, 1]
    L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')


    # 全连接层
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 10 * 64, 1024], stddev=0.1), name="W_fc1")
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name="b_fc1")
    h_pool2_flat = tf.reshape(L2_pool, [-1, 7*10*64])
    h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)


    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # readout层
    W_fc2 = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES], stddev=0.1), name="W_fc2")
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    return y_conv, keep_prob

def get_data(name, batch, if_random):
    tf_filename = os.path.join(FLAGS.tfrecord_dir, name + '.tfrecord')
    images, labels = wd.read_tfrecord(tf_filename,batch, if_random)
    #image_value, label_value = sess.run([images, labels])
    return images, labels

def add_train(input_images, input_labels):
    #train_tf = os.path.join(FLAGS.tfrecord_dir,'training.tfrecord')
    
    #train_images,train_labels = wd.read_tfrecord(train_tf, batch_size, True)
    #print("get images to train")
    #labels_d = tf.one_hot(train_labels, NUM_CLASSES, dtype = tf.float32)
    y_conv, keep_prob = characterClassificationNet(input_images)
    cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
        labels=input_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
    prediction = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(prediction, input_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    possibility = tf.nn.softmax(y_conv)
    return cross_entropy,train_step, keep_prob, accuracy, prediction, possibility
    
def get_image(test_image_path, sess):
    jpeg_data_tensor, decoded_image_tensor = wd.add_jpeg_decoding()
    images = wd.create_input_tensor(test_image_path, sess, jpeg_data_tensor, decoded_image_tensor)
    images = tf.reshape(images, [1, 40, 25, 1])
    image = sess.run(images)
    
    return image

def convert(img_list):
    gray_list = []
    for gray in img_list:
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray,(25, 40),interpolation=cv.INTER_LINEAR)
        cv.imwrite('temp.jpg', gray)
        cv.imshow('gray', gray)
        cv.waitKey(0)
        cv.destroyAllWindows()
        flag,gray = cv.threshold(gray,127,255,cv.THRESH_BINARY)           
        cv.normalize(src=gray, dst=gray,alpha=0, beta=1, norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
        gray_list.append(gray)
    gray_np = np.array(gray_list, dtype=np.float32)
    gray_np = gray_np.reshape([-1, 40, 25, 1])
    
    return gray_np
        
        
def class_word(img_list, cnn_var):
    prediction, keep_prob, image_tensor, sess = cnn_var
    img_np = convert(img_list)
    #test_image = get_image(os.path.join(FLAGS.testimage_dir, '4-label-53.jpg'), sess)
    prediction_value = sess.run(prediction, {keep_prob: 1.0, image_tensor: img_np})
    #tf.logging.info('prediction = %s' % (LETTERS_DIGITS[prediction_value[0]]))
    return prediction_value
        
        

    

def main(_):
    # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    image_tensor = tf.placeholder(tf.float32, [None, 40, 25, 1])
    label_tensor = tf.placeholder(tf.int64, [None,])
    cross_entropy,train_step, keep_prob, accuracy, prediction, possibility = add_train(image_tensor, label_tensor)
    #image_, label_ = get_data('training', batch_size, True)
   # image_valid, label_valid = get_data('validation', 156, True)
       
    #init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        
        #sess.run(init)
        saver.restore(sess, os.path.join(FLAGS.model_dir, 'model.ckpt'))
        cnn_var = [prediction, keep_prob, image_tensor, sess]
        #test_image = get_image(os.path.join(FLAGS.testimage_dir, '4-label-53.jpg'), sess)
        img = praseLabel2.cv_imread(r'F:\5.jpg')
        #region, region_neg = worddect.wordDetect(img, cnn_var)
        print(class_word([img], cnn_var))
        #prediction_value = sess.run(prediction, {keep_prob: 1.0, image_tensor: img_np})
        #tf.logging.info('prediction = %s, poss = %f' % (LETTERS_DIGITS[prediction_value[0]], possibility_value[0][prediction_value[0]]))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type = str,
        default = r'F:\licensePlateRecognition\data\training-set',
        help = 'Path to folders of images.'
    )
    parser.add_argument(
        '--valid_dir',
        type = str,
        default = r'F:\licensePlateRecognition\data\validation-set',
        help = 'Path to folders of images.'
    )
    parser.add_argument(
        '--testimage_dir',
        type = str,
        default = r'F:\licensePlateRecognition\data\images',
        help = 'Path to folders of labels.'
    )
    parser.add_argument(
        '--tfrecord_dir',
        type = str,
        default = r'F:\licensePlateRecognition\data\tf_records_67_7_3',
        help = 'Path to folder to save tfrecord.'
    )      
    parser.add_argument(
        '--model_dir',
        type = str,
        default = r'F:\licensePlateRecognition\save_67',
        help = 'Path to folder to save tfrecord.'
    )      
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    print(unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)        
