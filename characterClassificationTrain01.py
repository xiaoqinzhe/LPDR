# -*- coding: utf-8 -*-

import tensorflow as tf
'''
argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，
通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
'''
import argparse
import sys
import os
import time
import random
import numpy as np
from PIL import Image
from datetime import datetime

'''
导入其他写好的程序，只会导入函数，不会导入定义的变量
'''
import Word_data as wd
import praseLabel2


SIZE = 800  #40x20
WIDTH = 20
HEIGHT = 40
NUM_CLASSES = 72  #输入变量的个数
batch_size = 2048 #batch_size越大，训练迭代次数越少
epochs = 10000  #训练次数
FLAGS = None
#Keep_prob = 0.9
 
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D",
                  "E","F","G","H","J","K","L","M","N","P","Q","R","S","T",
                  "U","V","W","X","Y","Z","京","闽","粤","苏","沪","浙","皖",
                  "渝","甘","桂","贵","琼","冀","黑","豫","鄂","湘","赣","吉","辽",
                  "蒙","宁","青","鲁","晋","陕","川","津","新","云","藏","澳","港","挂","警","领","使","学")
license_num = ""


'''
建立的cnn网络
输入：x_image
输出：y_conv, keep_prob
'''

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



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

'''
得到图片和标签
'''
def get_data(name, batch, if_random):
    tf_filename = os.path.join(FLAGS.tfrecord_dir, name + '.tfrecord') #.tfrecord是后缀名
    images, labels = wd.read_tfrecord(tf_filename,batch, if_random) #调用word_data文件中的函数
    #image_value, label_value = sess.run([images, labels])
    return images, labels

'''
训练模型
传入参数：无
传出参数：交叉熵cross_entropy,训练步骤train_step, 保留率keep_prob
'''
def add_train(input_images, input_labels):
    #train_tf = os.path.join(FLAGS.tfrecord_dir,'training.tfrecord') #调用参数为tfrecord_dir的parser.add_argument()函数
    
    #train_images,train_labels = wd.read_tfrecord(train_tf, batch_size, True)  #类型为tensor
    #print("get images to train")
    #labels_d = tf.one_hot(train_labels, NUM_CLASSES, dtype = tf.float32)
    y_conv, keep_prob = characterClassificationNet(input_images) #调用建立的cnn网络
    cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
        labels=input_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
    prediction = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(prediction, input_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return cross_entropy,train_step, keep_prob, accuracy, prediction

'''
验证集不能再次调用cnn网络，全部在训练函数中得到
def add_valid():
    valid_tf = os.path.join(FLAGS.tfrecord_dir,'validation.tfrecord')
    valid_images,valid_labels = wd.read_tfrecord(valid_tf, 128, True)
    #labels_d = tf.one_hot(valid_labels, NUM_CLASSES, dtype = tf.float32)
    #再次调用cnn网络导致精确度很低
    y_conv, keep_prob = characterClassificationNet(valid_images)  
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), valid_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return  accuracy, keep_prob
    
def add_test(input_tensor):
    #test_images, test_labels = getTestPic()

    #images = tf.image.convert_image_dtype(images,tf.uint8)
    #images = tf.cast(images, tf.float32)
    
    y_conv, test_keep_prob = characterClassificationNet(input_tensor)
    #correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(test_labels, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return tf.argmax(y_conv, 1), test_keep_prob
'''
def get_image(test_image_path, sess):
    jpeg_data_tensor, decoded_image_tensor = wd.add_jpeg_decoding()
    images = wd.create_input_tensor(test_image_path, sess, jpeg_data_tensor, decoded_image_tensor)
    images = tf.reshape(images, [1, 40, 20, 1])
    
    

    

def main(_):
    # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
    '''
    reset_default_graph() ：用于清除默认图形堆栈并重置全局默认图形，只适用于当前线程
    '''
    tf.reset_default_graph()
    '''
    TensorFlow用五个不同级别的日志信息。为了升序的严重性，他们是调试DEBUG，信息INFO，警告WARN，错误ERROR和致命FATAL的。
    默认情况下，TensorFlow配置在日志记录级别的WARN，但当跟踪模型的训练，你会想要调整水平到INFO，这将提供额外的反馈如进程中的fit操作。
    '''
    tf.logging.set_verbosity(tf.logging.INFO)
    
    '''
    image:float32
    label:int64
    '''
    image_tensor = tf.placeholder(tf.float32, [None, 40, 20, 1])
    label_tensor = tf.placeholder(tf.int64, [None,])
    
    cross_entropy,train_step, keep_prob, accuracy, prediction = add_train(image_tensor, label_tensor)
    image_, label_ = get_data('training', batch_size, True)
    image_valid, label_valid = get_data('validation', 2048, True)       
    '''
    x = tf.placeholder(tf.float32, [1, 40, 25, 1])
    index, test_keep_prob = add_test(x) #(r'F:\licensePlateRecognition\data\images\0-label-4.jpg', sess)
    img = praseLabel2.cv_imread(r'F:\licensePlateRecognition\data\images\0-label-4.jpg')
    img = np.reshape(img, [1, 40, 25, 1])  
    img = img.astype(np.float32)、
    #tf.reset_default_graph()
    '''
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    '''
    GPU性能控制
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    '''
    只有运行了对话控制，才真正传入参数，开始进行训练
    '''
    with tf.Session(config=config) as sess:
        sess.run(init)
        #saver.restore(sess, os.path.join(FLAGS.model_dir, 'model.ckpt'))
        
        #index_value = sess.run(index, feed_dict={test_keep_prob: 1.0, x: img})
        #print(index_value)
        
        '''
        线程控制+队列
        '''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            '''
            i = 10
            valid_precise_value = sess.run(valid_precise,feed_dict={valid_keep_prob: 1.0})
            tf.logging.info('%s: Step %d: Validation accuracy = %f' % (datetime.now(), i, valid_precise_value))
            '''
            
            
            for i in range(epochs): 
                
                images, labels = sess.run([image_, label_])
                cross_entropy_value, _, accuracy_value = sess.run([cross_entropy, train_step, accuracy],
                                                  feed_dict={keep_prob: 0.5, image_tensor: images, label_tensor: labels})  
                if(i % 50 == 0):
                    tf.logging.info('%s: Step %d: Cross entropy = %f, Train_accuracy = %f' % 
                                    (datetime.now(), i, cross_entropy_value, accuracy_value))                
                if(i % 200 == 0):
                    
                    images, labels = sess.run([image_valid, label_valid])
                    valid_precise_value = sess.run(accuracy,feed_dict={keep_prob: 1.0, image_tensor: images, label_tensor: labels})
                    tf.logging.info('%s: Step %d: Validation accuracy = %f' % (datetime.now(), i, valid_precise_value))
            saver.save(sess, os.path.join(FLAGS.model_dir, 'model.ckpt'))
            
        except tf.errors.OutOfRangeError:
            print("out of range of threads")
        finally:
            coord.request_stop()
        
        coord.join(threads)  

#用这种方式保证了，如果此文件被其他文件import的时候，不会执行main中的代码
if __name__ == '__main__':
    parser = argparse.ArgumentParser()#创建 ArgumentParser() 对象
    '''
    add_argument() 方法添加参数，命令行参数可选
    
    ArgumentParser.add_argument(name or flags...[, action][, nargs][, const]
                                [, default][, type][, choices][, required][, help][, metavar][, dest])
    
    name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo。
    action - 命令行遇到参数时的动作，默认值是 store。
    store_const，表示赋值为const；
    append，将遇到的值存储成列表，也就是如果参数重复则会保存多个值;
    append_const，将参数规范中定义的一个值保存到一个列表；
    count，存储遇到的次数；此外，也可以继承 argparse.Action 自定义参数解析；
    nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号，当不指定值时对于 Positional argument 使用 default，对于 Optional argument 使用 const；或者是 * 号，表示 0 或多个参数；或者是 + 号表示 1 或多个参数。
    const - action 和 nargs 所需要的常量值。
    default - 不指定参数时的默认值。
    type - 命令行参数应该被转换成的类型。
    choices - 参数可允许的值的一个容器。
    required - 可选参数是否可以省略 (仅针对可选参数)。
    help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
    metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
    dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.
    
    '''
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
        default = r'F:\licensePlateRecognition\data\tf_records\tf_records_72_02',
        help = 'Path to folder to save tfrecord.'
    )      
    #保存模型
    parser.add_argument(
        '--model_dir',
        type = str,
        default = r'F:\licensePlateRecognition\saveModel\save_72_01', #保存目录
        help = 'Path to folder to save tfrecord.'
    )
    '''
    parse_args() 解析添加参数只能添加一个,如xx.py --image_dir 相应字符串
    parse_known_args()解析添加参数能添加多个，FLAGS保留第一个值，其余值保留在一个list中，如xx.py --image_dir 相应字符串 --testimage_dir 相应字符串
    '''      
    FLAGS, unparsed = parser.parse_known_args()
    #解析命令行参数，调用main函数 main(sys.argv)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)        
