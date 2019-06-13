# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:39:58 2018

@author: think
"""

import  xml.dom.minidom
import cv2 as cv
import numpy as np

'''
读取xml中tag之间的内容
'''
def getValue(root, tagName):
    tag = root.getElementsByTagName(tagName)
    content = tag[0].firstChild.data
    return content

'''
函数名  ：parseLabel
功能    ：读取标签文件，返回标签文件的内容，颜色和位置信息
输入参数：标签文件的path
返回值  ：data是字典类型，有[color, content, upL, upR, loL, upR]几个key
'''    
def parseLabel(path):
    #打开xml文档
    dom = xml.dom.minidom.parse(path)

    #得到文档元素对象
    root = dom.documentElement
    #获得颜色属性值
    color = getValue(root, 'name')
    #获得各个坐标值
    xmin = int(getValue(root, 'xmin'))
    xmax = int(getValue(root, 'xmax'))
    ymin = int(getValue(root, 'ymin'))
    ymax = int(getValue(root, 'ymax'))
    #获得内容属性值
    filename = getValue(root, 'filename')
    content = filename[0:-4]   
    data = {}
    data['color'] = color
    data['content'] = content
    data['lu'] = [xmin, ymin]
    data['rd'] = [xmax, ymax]
    
    return data

'''
CV_INTER_NN - 最近邻插值,
CV_INTER_LINEAR - 双线性插值 (缺省使用)
CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。
    当图像放大时，类似于 CV_INTER_NN方法
CV_INTER_CUBIC - 立方插值.
    '''
def zoomImg(image):
    shape = image.shape
    crop_img = []
    width = 1600
    height = 1200
    #根据图片长宽比对图片进行缩放，然后裁剪
    if(shape[0] / shape[1] >= 0.75):
        height = int(shape[0] * width / shape[1])
        res = cv.resize(image,(width,height),interpolation=cv.INTER_CUBIC)
        mid = height / 2
        crop_img = res[int(mid-600):int(mid+600), :]
    else:
        width = int(shape[1] * height / shape[0])
        res = cv.resize(image,(width,height),interpolation=cv.INTER_CUBIC)
        mid = width / 2
        crop_img = res[:, int(mid-800):int(mid+800)]
    
    return crop_img

"""
"""      
def drawRect(image, data):
    '''
    data[0,1,2,3,4] = [c, x, y, w, h]
    xmin =  data['upL'][0]
    xmax = data['loL'][0]
    ymin = data['upL'][1]
    ymax = data['upR'][1]
    '''
    xmin = int(data[1] - data[3] / 2)
    xmax = int(data[1] + data[3] / 2)
    ymin = int(data[2] - data[4] / 2)
    ymax = int(data[2] + data[4] / 2)
    #image = cv_imread(imgPath)
    #font=cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)
    cv.putText(image, str(data[0]), (xmin,ymin -20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    #image1 = cv.rectangle(image,(xmin,ymin-20),(xmin+20,ymin),(0,255,0),3)
    rect_img = cv.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),3)
    
    return rect_img

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
    cv.imencode('.jpg', img)[1].tofile(file_path)
    
if __name__=="__main__":
    imgPath = "藏A2838D.jpg"
    xmlPath = "藏A2838D.xml"    
    #读取图像，支持 bmp、jpg、png、tiff 等常用格式
    image = cv_imread(imgPath)
    #data = parseLabel(xmlPath)
    data = [0.886, 855, 887, 133, 40]
    crop_img = zoomImg(image)
    rect_img = drawRect(crop_img, data)
    cv_write('crop.jpg',crop_img)
    cv_write('藏-A6EE91_rect.jpg',rect_img)