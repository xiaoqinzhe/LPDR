# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:02:58 2018

@author: think
"""

import numpy as np
import cv2 as cv
import os  

import praseLabel2 as pl 
import nms
#import edgeDetectLocation as ed
import characterClassification_zq_v3 as ccy
    
flag = True
ffl = True

def search_rect_index_set(rect_index_set):
    count = 0
    for i in range(len(rect_index_set)):
        isMerge = False
        for j in range(i+1, len(rect_index_set)):
            if len(rect_index_set[j]) != 1:
                for m in range(len(rect_index_set[i])):
                    for n in range(len(rect_index_set[j])):
                        if(rect_index_set[i][m] == rect_index_set[j][n]):
                            isMerge = True
                            break
                if(isMerge == True):               
                    break
        if isMerge == True:
             count+=1
             rect_index_set[i] = list(set(rect_index_set[i] + rect_index_set[j]))
             rect_index_set[j] = [0]

    return count,rect_index_set

def distance(rect1, rect2):
    startX1, startY1, endX1, endY1 = rect1
    startX2, startY2, endX2, endY2 = rect2
    #计算中心向量差
    xc_diff = abs((startX2 + endX2 - startX1 - endX1) / 2)
    yc_diff = abs((startY2 + endY2 - startY1 - endY1) / 2)
    
    #因为这里的邻近应该是按边界来算，而不是中心点。因此，需要减去区域的长度
    xc_diff_pro = xc_diff - (endX1 - startX1) / 2 - (endX2 - startX2) / 2
    yc_diff_pro = yc_diff - (endY1 - startY1) / 2 - (endY2 - startY2) / 2
    
    return np.sqrt(np.square(max(xc_diff_pro, 0)) + np.square(max(yc_diff_pro, 0)))

def mergeChar(region, seed):
    output = []
    for i in range(len(region)):
        minX = 3000
        minY = 3000
        maxX = 0
        maxY = 0
        if len(region[i]) != 1:
            for j in range(len(region[i])):
                startX, startY, endX, endY = region[i][j]
                if(startX < minX): minX = startX
                if(startY < minY): minY = startY
                if(endX > maxX): maxX = endX
                if(endY > maxY): maxY = endY
            cv.rectangle(seed, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
            output.append([minX, minY, maxX, maxY])
            
    if flag == True:
        cv.namedWindow("seed", cv.WINDOW_NORMAL)
        cv.imshow("seed", seed)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return output   
    
def getNearChar(rect, char_probability, seed):
    rect_candidate = []
    for i in range(len(char_probability)):
        if(char_probability[i]!= 67):
             #cv.rectangle(seed, (rect[i][0], rect[i][1]), (rect[i][2], rect[i][3]), (0, 0, 255), 1)
             rect_candidate.append(rect[i])
    
    #找到距离相近的矩形         
    rect_index_set = []
    for i in range(len(rect_candidate)):
        tmp = []
        for j in range(i,len(rect_candidate)):
            h = rect_candidate[i][3] - rect_candidate[i][1]
            dis = distance(rect_candidate[i], rect_candidate[j])
            if dis < h*2 and abs(rect_candidate[i][1] - rect_candidate[j][1]) < h*0.7:
                tmp.append(j)
        rect_index_set.append(tmp)
    #print("距离相近的矩形序号")    
    #print(rect_index_set)    
    
    #合并距离相近的矩形，如[1, 3][2, 3]合并为[1, 2, 3]
    count = 1
    while count != 0:   
        count, rect_index_set = search_rect_index_set(rect_index_set)
    #print("归类之后的距离相近的矩形序号")
    #print(rect_index_set)   
    
    region = []
    for i in range(len(rect_index_set)):          
        tmp = []
        if len(rect_index_set[i]) != 1:
            for j in range(len(rect_index_set[i])):
                index = rect_index_set[i][j]
                tmp.append(rect_candidate[index])
            region.append(tmp)
    
    return region

#利用cnn判断各个候选字符是车牌字符的概率
def getCharProbability(img, pick, var):
    img_plate_set = []
    img_plate_neg_set = []
    for (startX, startY, endX, endY) in pick:
        #endY+=20
        #startY-=20
        if(startY < 0) : startY = 0 
        img_plate = img[startY:endY, startX:endX] 
        img_plate_set.append(img_plate)
        img_plate_neg = 255 - img_plate
        img_plate_neg_set.append(img_plate_neg)
    
    #用来判断白色字符    
    char_probability = ccy.class_word(img_plate_set, var)
    #对候选的灰度字符图像取反，用来判断黑色字符
    char_probability_neg = ccy.class_word(img_plate_neg_set, var)

    return char_probability, char_probability_neg

def mser(img, gray, var):
    #cv.imshow("gray", gray)
    #cv.imshow("grayneg", gray_neg)
    vis = img.copy()
    rect = img.copy()
    seed = img.copy()
    
    mser = cv.MSER_create(_max_area=2000)
    
    regions, _ = mser.detectRegions(gray)
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    #cv.polylines(img, hulls, 1, (0, 255, 0))
    keep = []
    #根据框的大小去掉那些不可能是字符的矩形
    for c in hulls:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(rect, (x, y), (x+w, y+h), (255, 255, 0), 1)
        #注：含有I的牌不容易找
        if h / w > 0.75 and h / w < 2.5:
            if h * w < 150 or h * w > 3000:
                continue
            keep.append([x, y, x + w, y + h])            
    #print("[x] %d initial bounding boxes" % (len(keep)))
    
    #利用非极大抑制算法将父子矩形合并，作为字符候选区域
    keep2 = np.array(keep)
    pick = nms.nms(keep2, 0.3)
    #print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
    for (startX, startY, endX, endY) in pick:
        cv.rectangle(vis, (startX, startY), (endX, endY), (0, 0, 255), 2)
    
    #获得各个字符候选的概率        
    char_probability, char_probability_neg = getCharProbability(img, pick, var)
    
    #利用候选字符的概率，将概率较大且相邻的候选字符归类
    region = getNearChar(pick, char_probability, seed)
    region_neg = getNearChar(pick, char_probability_neg, seed)

    #将所有找到归到同一类的候选字符连接起来
    result = mergeChar(region, seed)
    result_neg = mergeChar(region_neg, seed)
         
    if flag == True:
        #cv.imshow("rect", rect)
        cv.namedWindow("vis", cv.WINDOW_NORMAL)
        cv.imshow("vis", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return result, result_neg
       
def edgeDetect(img, region):
    plate_rect = []
    for (startX, startY, endX, endY) in region:
        width = endX - startX
        height = endY - startY
        if width / height < 1:
            continue
        minX = int(startX - width*1.5)
        minY = int(startY - height)
        maxX = int(endX + width*1.5)
        maxY = int(endY + height*0.5)
        if minX < 0: minX = 0
        if minY < 0: minY = 0
        if maxX > img.shape[0]: maxX = img.shape[0]
        if maxY > img.shape[1]: maxX = img.shape[1]
        
        plate_gray = cv.cvtColor(img[minY:maxY, minX:maxX], cv.COLOR_BGR2GRAY)
        output = ed.edgeDetect(plate_gray)
        #ed.drawRect(output, plate_gray)
        for box in output:        
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)
        
            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]
        
            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]
        
            plate_rect.append([x1+minX, y1+minY, x2+minX, y2+minY])
            
    if ffl == True:        
        plate = img.copy()
        for (startX, startY, endX, endY) in plate_rect:
            cv.rectangle(plate, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.namedWindow("plate", cv.WINDOW_NORMAL)
        cv.imshow("plate", plate)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return plate_rect

def wordDetect(img, var):    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    region, region_neg = mser(img, gray, var)
    
    return region, region_neg
 
if __name__=="__main__":
    if ffl == True:
        #灰度转换
        #取反值灰度  
        img = pl.cv_imread(r'E:\武大计算机实验室\智慧城市之车牌识别\数据集\copy\桂KL9551.jpg')
        region, region_neg = wordDetect(img)
        #plate_rect = edgeDetect(img, region)
        #print(plate_rect)
    else:
        imageFloder = r"E:\武大计算机实验室\智慧城市之车牌识别\数据集\copy"
        #listdir返回文件名的列表
        imageLine=os.listdir(imageFloder)
        
        iou_max = []
        count = 0
        #遍历整个列表
        for i in range(len(imageLine)):
            imgPath = imageFloder + '/' + imageLine[i]
            img = pl.cv_imread(imgPath)
            region, region_neg = wordDetect(img)
            #plate_rect = edgeDetect(img, region)
            img_copy = img.copy()
            if len(region) == 0 and len(region_neg) == 0:
                pl.cv_write("E:\\武大计算机实验室\\智慧城市之车牌识别\\数据集\\copyUnknow02\\"+imageLine[i], img_copy)
            else:
                
                for j in range(len(region)):
                    startX, startY, endX, endY = region[j]
                    cv.rectangle(img_copy, (startX, startY), (endX, endY), (0, 0, 255), 2)

                for j in range(len(region_neg)):
                    startX, startY, endX, endY = region_neg[j]
                    cv.rectangle(img_copy, (startX, startY), (endX, endY), (0, 255, 255), 2)
                pl.cv_write("E:\\武大计算机实验室\\智慧城市之车牌识别\\数据集\\copyPlate02\\"+imageLine[i], img_copy)
            
            #GenerateNegCases(region, img, imageLine[i])     
    