import cv2
import numpy as np
from PIL import Image
from iamge_operations import image_open,image_save

def harris_detection(image,kzise=5,k=0.06,threshold=619623619.2):#图像矩阵、区域大小、R检测参数和阈值
    print("Harris角点检测")
    image_matrix= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#将图片转换为灰度图
    height,width=image_matrix.shape
    I_x=cv2.Sobel(image_matrix,cv2.CV_64F, 1, 0, ksize=kzise)#求x\y方向导数
    I_y=cv2.Sobel(image_matrix,cv2.CV_64F, 0, 1, ksize=kzise)
    #构建M矩阵I准备
    I_x_square=np.multiply(I_x,I_x)
    I_y_square=np.multiply(I_y,I_y)
    I_xy=np.multiply(I_x,I_y)

    crf_matrix=np.zeros((height,width))#记录CRF值

    #遍历每个点，构建矩阵求解CRF值
    for x in range(kzise,height):
        for y in range(kzise,width):
            neighbor_x=I_x[x-2:x+3,y-2:y+3]
            neighbor_y=I_y[x-2:x+3,y-2:y+3]

            #存疑？计算特征值
            eigenvaluex=np.sum(neighbor_x)
            eigenvaluey=np.sum(neighbor_y)
            #计算角点响应函数值
            crf=eigenvaluex*eigenvaluey-k*((eigenvaluex+eigenvaluey)**2)
            
            if crf>threshold:#直接判断，小于阈值的CRF点不考虑
                crf_matrix[x,y]=crf
    #选取局部最大值作为候选点，阈值比较已做过了
    corner_points=np.zeros(image_matrix.shape)
    key_points=[]
    for x in range(kzise,height):
        for y in range(kzise,width):
            neighbors=crf_matrix[x-2:x+3,y-2:y+3]
            crf=crf_matrix[x,y]
            if crf_matrix[x,y]==neighbors.max() :
                corner_points[x,y]=crf #是角点
                #key_points.append(cv2.KeyPoint(x, y, 1))
            else:
                crf_matrix[x,y]=0
    
    for y in range(5, image.shape[0]):
        for x in range(5, image.shape[1]):
            if corner_points[y, x] != 0:
                key_points.append(cv2.KeyPoint(x, y, 1))
    key_points_image=cv2.drawKeypoints(image,key_points,image)
    print("Harris角点检测完毕")
    return key_points,key_points_image

