import cv2
import numpy as np

def matching_NNDR(key_points_1_descriptor,key_points_2_descriptor,key_points_1_pos,key_points_2_pos,threshold=0.8):
    print("NNDR匹配ing")
    best_matchs=[]
    best_matchs_pos=[]
    for x in range(len(key_points_1_descriptor)):
        ds=[]#所有距离
        best_match_index=0#最短距离的点的坐标记录
        min_d=np.inf
        for y in range(len(key_points_2_descriptor)):
            d=np.square(key_points_1_descriptor[x]-key_points_2_descriptor[y]).sum()
            ds.append(d)
            if d<min_d:#如果小于最短距离 就更新坐标和索引
                best_match_index=y
                min_d=d
        d_min=min(ds)
        ds.remove(d_min)
        d_2_min=min(ds)
        ratio_distance=d_min/d_2_min

        if ratio_distance<threshold:
            best_matchs.append(cv2.DMatch(x, best_match_index, 0))
            kp1_pos=key_points_1_pos[x]
            kp2_pos=key_points_2_pos[best_match_index]#得到匹配点的坐标
            pos=[kp1_pos,kp2_pos]
            best_matchs_pos.append(pos)
    print("NNDR匹配完成")
    return best_matchs,best_matchs_pos

#使用RANSAC算法优化匹配
def matching_RANSAC(matched_points_pos,max_iters=1000, epsilon=1):
    print("RANSAC优化中")
    best_matchs=[]
    size_=len(matched_points_pos)
    matched_points_pos=np.array(matched_points_pos,dtype="float32")
    #得到两幅图最佳匹配的分别坐标
    point_pos_1=matched_points_pos[:,0,:]
    point_pos_2=matched_points_pos[:,1,:]
    N=4#初始化的四个值
    #开始迭代
    for i in range(max_iters):
        #随机得到四个初始化样本
        matrix1=[]
        matrix2=[]
        pos1=[]
        pos2=[]
        indexs=np.random.randint(0,size_-1,N)#得到4个
        for j in range(N):
            index=indexs[j]
            pos1_x,pos1_y=matched_points_pos[index][0]
            pos2_x,pos2_y=matched_points_pos[index][1]
            pos1=[pos1_x,pos1_y]
            pos2=[pos2_x,pos2_y]
            matrix1.append(pos1)
            matrix2.append(pos2)
        matrix1=np.array(matrix1,dtype = "float32")
        matrix2=np.array(matrix2,dtype = "float32")
        #print(matrix1)
        #计算投影变换矩阵
        H=cv2.getPerspectiveTransform(matrix1,matrix2)
        #print(H)
        #计算每个点的拟合坐标
        Hp=cv2.perspectiveTransform(point_pos_1[None], H)[0]
        #检测内点
        minset=[]
        for i in range(size_):
            d=np.sum(np.square(point_pos_2[i] - Hp[i]))
            if d<epsilon:
                minset.append([point_pos_1[i],point_pos_2[i]])
        #print(len(minset))
        if len(minset) > len(best_matchs):
            best_matchs=minset
    print("优化完毕")
    return np.array(best_matchs)

def draw_matches(matches, img_left, img_right,path,thickness=1,verbose=False):
    height = max(img_left.shape[0], img_right.shape[0])
    width = img_left.shape[1] + img_right.shape[1]

    img_out = np.zeros((height, width, 3), dtype=np.uint8)
    img_out[0:img_left.shape[0], 0:img_left.shape[1], :] = img_left
    img_out[0:img_right.shape[0], img_left.shape[1]:, :] = img_right

    ow = img_left.shape[1]
   
    for p1,p2 in matches:
        p1o = (int(p1[1]), int(p1[0]))
        p2o = (int(p2[1] + ow), int(p2[0]))
        color = list(np.random.random(size=3) * 256)
        cv2.line(img_out, p1o, p2o, color, thickness=thickness)

    if verbose:
        print("Press enter to continue ... ")
        cv2.imshow("matches", img_out)
        cv2.waitKey(0)

    cv2.imwrite(path, img_out)