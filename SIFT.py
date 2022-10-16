import cv2
import numpy as np

def determine__direction(image_matrix):
    height,width=image_matrix.shape
    m_x_ys=np.zeros(image_matrix.shape)#模长
    theta_x_ys=np.zeros(image_matrix.shape)#大小
    #计算梯度值
    for x in range(1,height-1):
        for y in range(1,width-1):
            m=np.sqrt(np.square(image_matrix[x+1,y]-image_matrix[x-1,y])+np.square(image_matrix[x,y+1]-image_matrix[x,y-1]))
            theta=np.arctan2(image_matrix[x+1,y]-image_matrix[x-1,y],image_matrix[x,y+1]-image_matrix[x,y-1])
            theta=np.rad2deg(theta)#将弧度转换为角度
            m_x_ys[x,y]=m
            theta_x_ys[x,y]=theta
    return m_x_ys,theta_x_ys

def feature_description(image_matrix,key_points):
    print("产生SIFT算子")
    height,width=image_matrix.shape
    m_x_ys,theta_x_ys=determine__direction(image_matrix)
    #将关键点坐标提取出来，同时形成领域
    key_points_xy=[]
    descraptions=[]
    for key_point in key_points:
        descriptor = np.array(list())#描述子
        x=int(key_point.pt[0])
        y=int(key_point.pt[1])
        key_points_xy.append([x,y])#得到关键点坐标
        #关键点周围的邻域16*16
        neigbor_16=theta_x_ys[x-7:x+9,y-7:y+9]
        
        #划分4*4的种子区域
        for i in range(1,neigbor_16.shape[0],4):
            for j in range(1,neigbor_16.shape[1],4):
                seed=neigbor_16[i-1:i+3,j-1:j+3]#种子区域
                vector=np.zeros(8)
                for x1 in range(seed.shape[0]):
                    for y1 in range(seed.shape[1]):
                        orientation=seed[x1, y1]
                        index = int(orientation // 45)
                        vector[index]+=1          #得到一个统计图
                descriptor=np.concatenate((vector,descriptor), axis=None)
        if descriptor.shape[0]<128:
            descriptor=np.resize(descriptor,(128))
        #归一等操作
        normal=np.linalg.norm(descriptor)
        descriptor=np.divide(descriptor, normal)
        threshold_descriptor=np.array(descriptor)

        for i in range(descriptor.shape[0]):
            if descriptor[i] < 0.2:
                threshold_descriptor[i]=descriptor[i]
            else:
                threshold_descriptor[i]=0

        normal_again = np.linalg.norm(threshold_descriptor)
        threshold_descriptor = np.divide(threshold_descriptor, normal_again)
        descraptions.append(threshold_descriptor)
    print("产生SIFT算子完成")
    return descraptions,key_points_xy#返回描述子和坐标
