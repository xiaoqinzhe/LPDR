# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:18:25 2018

@author: 15827
"""

import cv2 as cv
import os
import numpy as np
import random
import string
from PIL import Image

height = 40
width = 20

'''
给图像随机加白点, 255是白色
'''
def salt(img, n):    
    for k in range(n):
        i = int(np.random.random() * img.shape[1]) #产生随机数（0~1）*图像的宽
        j = int(np.random.random() * img.shape[0]) #产生随机数（0~1）*图像的高
        if img.ndim == 2:      #如果图像是二维的
            img[j,i] = 255    
        elif img.ndim == 3:    #如果图像是三维的，分别给三个通道的对应像素点赋值
            img[j,i,0]= 255    
            img[j,i,1]= 255    
            img[j,i,2]= 255    
    return img

'''
给图像随机加黑点, 0是黑色
'''
def pepper(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] == 0
        elif img.ndim == 3:
            img[j,i,0]= 0    
            img[j,i,1]= 0    
            img[j,i,2]= 0
    return img 

'''
图片旋转
'''
def warp(img, angel):
    rows, cols=img.shape[:2] #切片，rows=图片高,cols=图片宽
    '''
    cv.getRotationMatrix2D(旋转的中心点,旋转的角度,图像缩放因子)
    获得图像绕着某一点的旋转矩阵
    
    '''
    M=cv.getRotationMatrix2D((cols/2,rows/2),angel,0.6)
    '''
    cv.warpAffine(旋转图片,旋转矩阵,变换后的图片大小)
    对图像做仿射变换
    '''
    dst=cv.warpAffine(img,M,(cols,rows)) #图片旋转后大小不变
    return dst

def warpA(img):
    rows,cols=img.shape[:2]
    r1 = np.random.randint(0, 1)
    r2 = np.random.randint(0, 1)
    r3 = np.random.randint(0, 1)
    pts1=np.float32([[10,10],[10,30],[20,10]])
    pts2=np.float32([[10,10],[10,30],[20,10]]) + np.float32([[r1], [r2], [r3]])
    M=cv.getAffineTransform(pts1,pts2)
    dst=cv.warpAffine(img,M,(cols,rows))
    return dst

'''
图片透视
'''
def perspect(img):
    rows,cols=img.shape[:2]
    r1 = np.random.randint(0, 1.5)
    r2 = np.random.randint(0, 1.5)
    r3 = np.random.randint(0, 1.5)
    r4 = np.random.randint(0, 1.5)
    '''
    cv.getRotationMatrix2D(pts需要变换前后的4个点对应位置)
    两个位置就是变换前后的对应位置关系
    '''
    pts1=np.float32([[6,10],[15,11],[7,29],[16,31]])
    pts2=np.float32([[6,10],[15,11],[7,29],[16,31]])+ np.float32([[r1], [r2], [r3], [r4]])
    M=cv.getPerspectiveTransform(pts1,pts2)
    '''
    cv.warpPerspective(透视图片,透视矩阵,变换后的图片大小)
    对图像做仿射变换
    '''
    dst=cv.warpPerspective(img,M,(cols,rows))
    return dst

'''
生成文件夹
'''
def mkdir(path,num):
    for i in range(num):
        writeDir = path + '\\'+ str(i)
        isExists=os.path.exists(writeDir)
        if not isExists:
            os.mkdir(writeDir)
    


'''
图片保存随机命名
'''
def save_image(save_path, img, filename=None):
    if filename is None:        
        image_name = os.path.join(save_path,shullf_name(16) + '.png')
        while os.path.exists(image_name):  #如果随机生成的文件名存在，则重新继续生成文件
            image_name = os.path.join(save_path,shullf_name(16) + '.png')
    else:
        image_name = os.path.join(save_path, filename) 
    return image_name

def shullf_name(Num):
    
    return ''.join(random.choice(string.ascii_uppercase + 
                                 string.ascii_lowercase+ string.digits) for _ in range(Num))
 
"""
解决OpenCV的imread函数无法读取中文路径和中文命名的文件的问题
"""
def cv_imread(file_path):
    img_mat = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    return img_mat

"""
解决OpenCV的imwrite函数无法写入中文路径和中文命名的文件的问题
"""
def cv_write(file_path, img):
    cv.imencode('.png', img)[1].tofile(file_path)

'''
灰度化+二值化+裁剪
'''
def grayBinary(img):
    grey_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  #灰度化
    flag,binary_img = cv.threshold(grey_img,127,255,cv.THRESH_BINARY)  #二值化
    binary_img = cv.resize(binary_img,(20, 40),interpolation=cv.INTER_LINEAR)  #线性裁剪
    #binary_img = cv.resize(img,(20, 40),interpolation=cv.INTER_LINEAR)
    return binary_img



'''
移动(左右 和 上下)
'''
def crop_image(img, crop_ratio = 3):
    row, col = img.shape
    hori_left = int(random.randint(0, crop_ratio) / 100 * col)
    hori_right = int((1.0 - random.randint(0, crop_ratio) / 100) * col)
    vert_up = int(random.randint(0, crop_ratio * 2) / 100 * row) #图片比例2:1，因此垂直线上黑空更多
    vert_down = int((1.0 - random.randint(0, crop_ratio * 2) / 100) * row)
    #print(hori_left, hori_right, vert_up, vert_down)
    
    crop_img = img[vert_up:vert_down, hori_left:hori_right]  #同时进行上下左右平移
    
    #crop_img = img[10:height, 0:width]  #上平移
    #crop_img = img[0:30, 0:width]  #下平移
    #crop_img = img[0:height, 5:width]  #左平移
    #crop_img = img[0:height, 0:15]  #右平移
    
    re_crop = cv.resize(crop_img, (col, row))
    
    return re_crop


'''
腐蚀
'''
def erosion(img,random):
    kernel = np.ones((random,1),np.uint8) 
    erosion = cv.erode(img,kernel,iterations = 1)  #腐蚀,iterations表示
    return erosion
    
'''
膨胀
'''
def dilate(img,random):
    kernel = np.ones((random,1),np.uint8) 
    dilation = cv.dilate(img,kernel,iterations = 1) #膨胀
    return dilation
    
    

'''
cv.imshow('temp', init_image)   #显示图像
cv.waitKey(0)  #等待按键结束
cv.destroyAllWindows()   #销毁窗口释放内存
'''

'''
像素取反
'''
def pixelInversion(img):
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i,j]= 255 - img[i,j]
    return img

'''
在图片特定位置加白色噪声
'''
def crt(step, img):
    vis = img.copy()
    w = 14
    h = 35
    width = w
    height = h
    w_diff = 10
    h_diff = 2
    for e in range(step):
        f = np.random.randint(0,3)
        if f==0:
            w = w - 1
        if f==1:
            h = h + 1
        if f==2:
            w = w + 1
        if f==3:
            h = h - 1
            
        if height - h >= h_diff or h < 1:
            h = h + 1
        if h - height >= h_diff or h >= img.shape[0]:
            h = h - 1
        if width - w >= w_diff or w < 1:
            w = w + 1
        if w - width >= w_diff or w >= img.shape[1]:
            w = w - 1
        vis[h,w] = 255
    
    return vis    


   
if __name__ == '__main__':
    '''
    读取图片，目录格式斜杠可以为//或者为r\
    '''
    readDir = r'F:\licensePlateRecognition\data\1'
    writeDir = r'F:\licensePlateRecognition\data\2'
    files = os.listdir(readDir)
    #mkdir(writeDir,65) #生成目录
    #img = cv_imread(r'F:\licensePlateRecognition\data\20x40训练集和验证集(未加噪800)\training-set\4\2nBHX0RE9ZaNNX4k.png')
    #last = erosion(img)
    #temp1 = np.random.randint(0, 7)
    #last = dilate(img,temp1)
    #cv.imshow('temp', last)   #显示图像
    #cv.waitKey(0)  #等待按键结束
    #cv.destroyAllWindows()   #销毁窗口释放内存
    '''
    HSV模型：色调（H），饱和度（S），明度（V）
    '''
	 #HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV) #RGB转HSV
	 #h, s, v = cv.split(HSV)  #分割hsv值
	 #re_image = cv.resize(gray, (40, 32))  #图像缩放

	 #BRG = cv.cvtColor(HSV, cv.COLOR_HSV2BGR)  #HSV转RGB
	 #img = salt(img, 100)
	 #img = pepper(img, 100)
	 #img = warp(img, 45)#旋转
	 #img = warpA(img)
	 #cv.imshow('img', img)
	 #cv.waitKey(0)
	 #praseLabel2.cv_write(os.path.join(save_path, '港.jpg'), BRG)  #写入文件
     
    for subfile in files:
        '''
        os.path.join(A,B,C):结合目录路径A\B\C
        os.listdir():返回指定的文件夹包含的文件和文件夹的'名字'的列表
        
        '''
        file_image = os.listdir(os.path.join(readDir, subfile))
        '''
        每次取文件夹中前j个文件，进行加噪得到i个处理后的文件
        '''
        for j in range(34):
            #if len(file_image) > 0: #文件个数大于0
            readImgDir = os.path.join(readDir, subfile, file_image[j])
            img = cv_imread(readImgDir)
            #last = grayBinary(img)
            #writeImgDir = os.path.join(writeDir,subfile) #将读取的目录和写入的根目录结合得到写入目录
            #save_name = save_image(writeImgDir,last) #随机生成文件
            #cv_write(save_name, last)  #(文件名，文件)
            for i in range(10):
                #last = grayBinary(img)
                #temp1 = np.random.randint(0,1)
                #temp1 = np.random.randint(4,8)
                temp1 = 1
                temp2 = np.random.randint(0, 10) #返回随机的整数，位于半开区间 [low, high)。
                temp3 =  np.random.randint(0, 30)
                temp4 = np.random.randint(-5,5)
                last = salt(img, temp2)
                #last = pepper(img, temp3)
                #last = warpA(img)
                #last = perspect(img)
                #last =  crop_image(img)
                #last = pixelInversion(img)
                #last = crt(500, img)
                #last = erosion(img,temp1)
                #last = dilate(img,temp1)
                '''
                if i %2 == 0 :
                    last = warpA(last)
                else:
                    last = perspect(last)
                '''
                writeImgDir = os.path.join(writeDir,subfile) #将读取的目录和写入的根目录结合得到写入目录
                save_name = save_image(writeImgDir,last) #随机生成文件名
                cv_write(save_name, last)  #(文件名，文件)
    


