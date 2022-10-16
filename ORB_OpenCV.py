#使用算法包进行特征检测和匹配
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

file_='pic1'
picture1_path="data\pic1\Yosemite1.jpg"
picture2_path="data\pic1\Yosemite2.jpg"
key_point_image1_path="data"+"\\"+file_+"\\"+"kp1_image_SIFT.jpg"
key_point_image2_path="data"+"\\"+file_+"\\"+"kp2_image_SIFT.jpg"
match_NNDR_path="data"+"\\"+file_+"\\"+"Matches_NNDR.png"
match_NNDR_RANSAC_path="data"+"\\"+file_+"\\"+"Matches_RANSAC.png"


img1 = cv.imread(picture1_path,0)          # queryImage
img2 = cv.imread(picture2_path,0) # trainImage

# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches, None, flags=2)
plt.imshow(img3),plt.show()




