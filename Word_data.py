# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 08:57:34 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:25:14 2018
data
@author: Zhang Xiaopiao
"""

import tensorflow as tf
import numpy as np

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import cv2
import praseLabel2
FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
def test_flag():
    print(FLAGS.image_dir)
def create_image_lists(train_dir, valid_dir):
    """Building a list of training images from the file system.
    
    Splits image into stable training,validation sets,and return a 
    data structure describing the lists of images for each label and 
    their paths.
    
    Args:
        image_dir: String path to a folder including images and labels
        test_dir: Usually testimages and validation is from same dir,but not this time
        validation_percentage: Integer percentage of the images to reserve for tests.(<= 40)
    Returns:
        A dictionary containing an entry for train_images,testing_images,validation_images
    """
    if not tf.gfile.Exists(train_dir):
        tf.logging.error("Image directory '" + train_dir + "' not found.")
        return None
    if not tf.gfile.Exists(valid_dir):
        tf.logging.error("Image directory '" + valid_dir + "' not found.")
        return None
    extensions = ['jpg', 'jpeg','bmp','png']
    #file_list = []
    train_images = {}
    validation_images = {}
    #testing_images = []
    
    labels = os.listdir(train_dir)
    for label_name in labels:
        train_images[label_name] = []
        for extension in extensions:
            file_glob = os.path.join(train_dir,label_name, '*.' + extension)
            train_images[label_name].extend(tf.gfile.Glob(file_glob))

    labels = os.listdir(valid_dir)
    for label_name in labels:
        validation_images[label_name] = []
        for extension in extensions:
            file_glob = os.path.join(valid_dir,label_name, '*.' + extension)
            validation_images[label_name].extend(tf.gfile.Glob(file_glob))
            
        result = {
            'training' : train_images,
            'validation' : validation_images
        }
    return result
    
def add_jpeg_decoding():
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Returns:
        Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    From Tensorflow!
    """
    input_height, input_width, input_depth = (40, 20, 1)
    jpeg_data = tf.placeholder(tf.string, name = 'JPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
    decode_image = tf.squeeze(resized_image,[0])
    #image_int = tf.image.convert_image_dtype(decode_image,tf.uint8)
    return jpeg_data,decode_image

def create_input_tensor(image_path, sess, jpeg_data_tensor, decoded_image_tensor):
    """Create a lists input_array file for cache
    
    Arg: 
        image_path: the complete path of a image
        sess: the tensorflow session
        jpeg_data_tensor: raw iamge tensor placeholder
        decoded_image_tensor: the decoded tensor placeholder including the options to decode raw image
    Returns:
        A array of decoded image with confirmed shape(1 * 1440 * 1080 * 3)
    """
    #tf.logging.info('Creating input_tensor at ' + save_path)
    #for image_path in image_lists:
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    input_array = sess.run(decoded_image_tensor, 
                            {jpeg_data_tensor:image_data})
    return input_array
           
        

def create_tfrecord(result, sess, tf_path):
    """
    create tfrecord files for train,validation,test
    Args:
        path: the dictionary of tfrecords
        result: the dictionary of images
        sess: the session
    Return: None
    """
    #path = FLAGS.tfrecord_dir
    count = 0
    if not tf.gfile.Exists(tf_path):
        tf.gfile.MakeDirs(tf_path)
    catelogue = ['training', 'validation']  #['training','testing','validation']        
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding()
    for data_name in catelogue:
        tf_filename = os.path.join(tf_path,data_name + '.tfrecord')
        writer = tf.python_io.TFRecordWriter(tf_filename)
        labels = result[data_name].keys()
        for label in labels:
            image_files = result[data_name][label]
         
            for index_val,file in enumerate(image_files):
                tf.logging.info("write the %dst in %s"%(index_val,data_name))
                #name = os.path.basename(file)
                #name, _ = os.path.splitext(name) 
                #label= get_label(label_path, name + '.xml')
                input_image_array = create_input_tensor(file, sess, jpeg_data_tensor, decoded_image_tensor)
                input_image_string = input_image_array.tostring()
                label_int = np.int64(label)
                #label_string = (np.int64(label)).tostring()
                example = tf.train.Example(features = tf.train.Features(
                        feature = {
                                'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label_int])),
                                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [input_image_string]))
                                }))
                writer.write(example.SerializeToString())
                count += 1
        writer.close()
        
    print("count = ", count)

def read_tfrecord(file_name,batch,is_random):
    #file_name = os.path.join(FLAGS.)
    filename_queue = tf.train.string_input_producer([file_name])
    '''
    tf.train.string_input_producer([file_name],num_epochs)
    这里有个参数是num_epochs，指定好之后，Tensorflow自然知道如何读取数据，
    保证在遍历数据集的一个epoch中样本不会重复，也知道数据读取何时应该停止。
    '''
    reader = tf.TFRecordReader()
    #tfrecord_shape = filename_queue.size()
    _, serialize_example = reader.read(filename_queue)
    feature = tf.parse_single_example(serialize_example,
                                       features = {
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image': tf.FixedLenFeature([], tf.string),
                                               })
    
    labels = tf.cast(feature['label'], tf.int64)
    
    #labels = tf.reshape(labels, [1])
    #labels = tf.squeeze(labels)
    images = tf.decode_raw(feature['image'], tf.float32)
    #print(images.shape)
    images = tf.reshape(images, [40, 20, 1])
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    #images = tf.squeeze(images)
    

    
    #images = tf.image.convert_image_dtype(images,tf.uint8)
 
    #images = images > 190
    #images = tf.cast(images, tf.float32)
    
    if is_random == True:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                batch_size=batch,
                                                capacity=3000,
                                                num_threads=2,
                                                min_after_dequeue=200)
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                        batch_size=batch,
                                        capacity=200,
                                        num_threads=1)        
                                                
    return images_batch, labels_batch
    """
    for i in range(2):
        image,label = sess.run([images,labels])
        image = tf.squeeze(image)
        image = tf.cast(image,tf.int64)
        cv2.imshow('image' + str(i),image)
    """
    #coord.request_stop()
    #coord.join(threads)
    
    
      
    


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    result = create_image_lists(FLAGS.image_dir, FLAGS.valid_dir)
    
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        #image = tf.reverse(image, [3])
        create_tfrecord(result, sess, FLAGS.tfrecord_dir)
        '''
        train_tf = os.path.join(FLAGS.tfrecord_dir,'validation.tfrecord')
        
        image,label= read_tfrecord(train_tf,512,True)
        image = tf.image.convert_image_dtype(image,tf.uint8)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        try:
            for j in range(10):
                img, labe= sess.run([image, label])
                #print(img.shape, labe.shape)
               # License_plate_localizition.test_save_image(img, labe, 
                                        #r'G:\GraduateStudy\智慧城市\Task3_车牌识别\来吉测试车牌识别图片\Test_image')
                
                for i in range(12):
    
                    name = str(i)+'-label-'+ str(labe[i]) + '.jpg'
                    path = os.path.join(FLAGS.testimage_dir, name)
                    #print(img[i])          
                    praseLabel2.cv_write(path,img[i])             
                    #print(labe[i])
            
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()
            
        coord.join(threads)
    '''
     
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
        default = r'F:\licensePlateRecognition\data\temp_test',
        help = 'Path to folders of labels.'
    )
    parser.add_argument(
        '--tfrecord_dir',
        type = str,
        default = r'F:\licensePlateRecognition\data\tf_records\tf_records_67_01',
        help = 'Path to folder to save tfrecord.'
    )      
    parser.add_argument(
        '--num_grid',
        type = int,
        default = 5,
        help = 'numbers of grid per image'
    )       
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)        
        
