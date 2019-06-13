# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:51:14 2018

@author: 15827
"""
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
import decode_excel as de

SIZE = 1000  #32x40
WIDTH = 20
HEIGHT = 40
NUM_CLASSES = 72  #输入变量的个数
batch_size = 2048
epochs = 20000
FLAGS = None
#Keep_prob = 0.9
 
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D",
                  "E","F","G","H","J","K","L","M","N","P","Q","R","S","T",
                  "U","V","W","X","Y","Z","京","闽","粤","苏","沪","浙","皖",
                  "渝","甘","桂","贵","琼","冀","黑","豫","鄂","湘","赣","吉","辽",
                  "蒙","宁","青","鲁","晋","陕","川","津","新","云","藏","澳","港","挂","警","领","使","学")
license_num = ""


def characterClassificationNet(x_image):
    #卷积层1+池化层1
    #W_conv1 = weight_variable([5, 5, 1, 32])
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    #b_conv1 = bias_variable([32])
    b_conv1 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    #卷积层2+池化层2
    #W_conv2 = weight_variable([5, 5, 32, 64])
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    #b_conv2 = bias_variable([64])
    b_conv2 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    #输出层前加上防过拟合
    keep_prob = tf.placeholder(tf.float32)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)
    
    
    #全连接层1
    #W_fc1 = weight_variable([5 * 10 * 64, 1024])
    W_fc1 = tf.Variable(tf.truncated_normal([5 * 10 * 64, 1024], stddev=0.1))
    #b_fc1 = bias_variable([1024])
    b_fc1 = tf.Variable(tf.truncated_normal([1024], stddev=0.1))
    h_pool2_flat = tf.reshape(h_pool2_drop, [-1, 5 * 10 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #全连接层2
    #W_fc1 = weight_variable([5 * 10 * 64, 1024])
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
    #b_fc1 = bias_variable([1024])
    b_fc2 = tf.Variable(tf.truncated_normal([512], stddev=0.1))
    h_pool3_flat = tf.reshape(h_fc1, [-1, 1024])
    h_fc2 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc2) + b_fc2)
    
    #输出层前加上防过拟合
    #keep_prob = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    
    #输出层
    #W_fc2 = weight_variable([1024, NUM_CLASSES])
    W_fc3 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1))
    #b_fc2 = bias_variable([NUM_CLASSES])
    b_fc3 = tf.Variable(tf.truncated_normal([NUM_CLASSES], stddev=0.1))
    
    y_conv=tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    
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
    images = tf.reshape(images, [1, 40, 20, 1])
    image = sess.run(images)
    
    return image

def convert(img_list):
    gray_list = []
    for img in img_list:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray,(20, 40),interpolation=cv.INTER_LINEAR)
        flag,gray = cv.threshold(gray,127,255,cv.THRESH_BINARY)           
        cv.normalize(src=gray, dst=gray,alpha=0, beta=1, norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
        gray_list.append(gray)
    gray_np = np.array(gray_list, dtype=np.float32)
    gray_np = gray_np.reshape([-1, 40, 20, 1])
    
    return gray_np

'''
读取多张图片
'''
def convertT(imgDir,files):
    gray_list = []
    for i in range(len(os.listdir(imgDir))):
        file_image = os.listdir(imgDir)
        img_path = os.path.join(imgDir,file_image[i])
        img = cv_imread(img_path)
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.resize(img,(20, 40),interpolation=cv.INTER_LINEAR) #图片裁剪
        flag,gray = cv.threshold(gray,127,255,cv.THRESH_BINARY)           
        cv.normalize(src=gray, dst=gray,alpha=0, beta=1, norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
        gray_list.append(gray)
    gray_np = np.array(gray_list, dtype=np.float32)
    gray_np = gray_np.reshape([-1, 40, 20, 1])
    #print(gray_np)
    
    return gray_np,files
        
       
        
def class_word(img_list, cnn_var):
    prediction, keep_prob, image_tensor, sess = cnn_var
    img_np = convert(img_list)
    #test_image = get_image(os.path.join(FLAGS.testimage_dir, '4-label-53.jpg'), sess)
    prediction_value = sess.run(prediction, {keep_prob: 1.0, image_tensor: img_np})
    #tf.logging.info('prediction = %s' % (LETTERS_DIGITS[prediction_value[0]]))
    return prediction_value

def class_word_T(img_list, cnn_var,files):
    prediction, keep_prob, image_tensor, sess = cnn_var
    img_np,img_name = convertT(img_list,files)
    #test_image = get_image(os.path.join(FLAGS.testimage_dir, '4-label-53.jpg'), sess)
    prediction_value = sess.run(prediction, {keep_prob: 1.0, image_tensor: img_np})
    #print(prediction_value,img_name)
    #tf.logging.info('prediction = %s' % (LETTERS_DIGITS[prediction_value[0]]))
    return prediction_value,img_name
        
     
def load_Img(imgDir,imgFoldName):
     imgs = os.listdir(imgDir+imgFoldName)
     imgNum = len(imgs)
     data = np.empty((imgNum,1,12,12),dtype="float32")
     label = np.empty((imgNum,),dtype="uint8")
     for i in range (imgNum):
         img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
         arr = np.asarray(img,dtype="float32")
         data[i,:,:,:] = arr
         label[i] = int(imgs[i].split('.')[0])
     return data,label   

"""
解决OpenCV的imread函数无法读取中文路径和中文命名的文件的问题
"""
def cv_imread(file_path):
    img_mat = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    return img_mat
def gene_dict():
    right_dict = {}
    wrong_dict = {}
    for i in LETTERS_DIGITS:
        right_dict[i] = 0
        wrong_dict[i] = 0
        
    return right_dict, wrong_dict



def main(_):
    
    accuracy_num = 0
    img_list = r'F:\licensePlateRecognition\data\test-set\testing-set2'  #str类型
    files = os.listdir(img_list)

    
    # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
    tf.reset_default_graph() #第一次运行程序时需要加上，否则会出现not found in checkpoint情况
    tf.logging.set_verbosity(tf.logging.INFO)
    image_tensor = tf.placeholder(tf.float32, [None, 40, 20, 1])
    label_tensor = tf.placeholder(tf.int64, [None,])
    cross_entropy,train_step, keep_prob, accuracy, prediction, possibility = add_train(image_tensor, label_tensor)
    #image_, label_ = get_data('training', batch_size, True)
    # image_valid, label_valid = get_data('validation', 156, True)
       
    #init = tf.global_variables_initializer()   #测试程序不能初始化
    paths,plate_list = de.cons_gt(r'F:\licensePlateRecognition\data\test-set\img-test')
    saver = tf.train.Saver()
    right_dict, wrong_dict = gene_dict()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    count = 0
    with tf.Session(config=config) as sess:
        
        #sess.run(init)
        '''
        导入训练好的cnn网络模型
        '''
        saver.restore(sess, os.path.join(FLAGS.model_dir, 'model.ckpt'))
        cnn_var = [prediction, keep_prob, image_tensor, sess]
        
        for i in range(len(files)):
            test_name = []
            Img_name = []
            imgDir = os.path.join(img_list,files[i])
            #test_prediction,Img_name = class_word_T(imgDir, cnn_var,files[i])
            test_prediction,_ = class_word_T(imgDir, cnn_var,files[i])
            Img_name = plate_list[int(files[i])]
            #print(test_prediction)
            
            #if len(test_prediction) < 7:
                #count = count + 1
                
            if (len(test_prediction)) < 7:
                continue
            for j in range(len(test_prediction)):
                test_name.append(LETTERS_DIGITS[test_prediction[j]])
                
            test_name = "".join(test_name)
            #Img_name = "".join(Img_name)
            #print(test_name)
            #print(Img_name)
            #print('-------------------')
            #test_name = list(test_name)
            #Img_name = list(Img_name)
            #number = 0S
            #if len(test_name) == 7:
                #for k in range(len(test_name)):
                    #if test_name[k] == Img_name[k] and Img_name[k] == '1':
                        #print(test_name[k])
                        #print(Img_name[k])
                        #count = count + 1
                        #print('------------')
                        #number =number + 1
                #print(number)
                #if number == 7:
                    #count = count + 1
        #print(count)
            
            
                    
                        
            
            #print(test_name)
            #print(Img_name)
            #print(type(test_name),type(Img_name))
            count += 1
            if test_name == Img_name:
                accuracy_num=accuracy_num+1
            else:
                print(imgDir)
                print(test_name)
                print(Img_name)
                print('--------------------')
            for pred, char in zip(test_name, Img_name):
                if char == pred:
                    right_dict[char] += 1
                else:
                    #print(pred)
                    #print(char)
                    #print(Img_name)
                    wrong_dict[char] += 1
                    #print('------------------')
                #print("预测正确")
            #else:
                #print("预测错误")
                
                
            #if len(test_prediction)>=7 and test_name != Img_name:
                #print(test_name)
                #print(Img_name)
                #print('----------------')
            
            
                      
                

            #print("----------------------------------")
        accuracy_pro = accuracy_num/count
        print(len(files),count, accuracy_num,accuracy_pro)
        sorted_right_dict = sorted(right_dict.items(),key=lambda ab:ab[1],reverse=True)
        sorted_wrong_dict = sorted(wrong_dict.items(), key=lambda a:a[1], reverse=True)
        print('right :', sorted_right_dict)
        print('########################')
        print('wrong :', sorted_wrong_dict)
        #print(count)
        
        
        
        
        #test_image = get_image(os.path.join(FLAGS.testimage_dir, '4-label-53.jpg'), sess)

        #img = praseLabel2.cv_imread(r'C:\Users\Administrator\Desktop\copy\浙BB5655.jpg')
        #region, region_neg = worddect.wordDetect(img, cnn_var)
        
        
        #prediction_value = sess.run(prediction, {keep_prob: 1.0, image_tensor: img_np})
        #tf.logging.info('prediction = %s, poss = %f' % (LETTERS_DIGITS[prediction_value[0]], possibility_value[0][prediction_value[0]]))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type = str,
        default = r'F:\licensePlateRecognition\data\training-set1',
        help = 'Path to folders of images.'
    )
    parser.add_argument(
        '--valid_dir',
        type = str,
        default = r'F:\licensePlateRecognition\data\validation-set1',
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
        default = r'F:\licensePlateRecognition\data\tf_records\tf_records_72_02',
        help = 'Path to folder to save tfrecord.'
    )      
    parser.add_argument(
        '--model_dir',
        type = str,
        default = r'F:\licensePlateRecognition\saveModel\save_72_03',
        help = 'Path to folder to save tfrecord.'
    )      
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)        
