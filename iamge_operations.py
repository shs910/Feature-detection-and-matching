import numpy as np
from PIL import Image
import cv2

#根据路径打开图像直接使用cv2的函数imread,得到Numpy数组格式,但是需要改变数据格式为float才可以用于计算

#可使用cv2代替PIL
def image_open(image_path):
    image=Image.open(image_path)#以RGB新式打开数据，若使用CV2，需要转换下
    image=np.array(image,dtype=np.float32)#转化成浮点数用于后续操作
    return image

def image_rescale(image):#改变图像大小（归一化处理？利于保存）
    image = image.astype('float32')
    now_max=np.max(image)
    now_min=np.min(image)
    image=(image-now_min)/(now_max-now_min)*255
    return image

def image_save(image,store_path):
    image=image_rescale(image)
    image = np.array(image, dtype = np.uint8)#转化成nint8整数格式存储
    cv2.imwrite(store_path,image)

def load_all_images_2(file_name):
    with open(file_name, 'r') as file:
        number_of_files = int(file.readline().rstrip())
        images_paths = []
        for i in range(number_of_files):
            images_paths.append(file.readline().rstrip())
        images_mask = file.readline().rstrip()
    
    images = []
    for image_path in images_paths:
        images.append(image_open(image_path))

    mask_image=image_open(images_mask)
    mask_image=cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    print("加载成功")
    return images, mask_image
#可改：使用os代替txt文件
def load_all_images(txt_file_path):
    with open(txt_file_path,'r') as file:#使用只读模式打开存有各图片命（路径）的txt文件？是否使用os库操作
        #加载图片路径
        image_nums=int(file.readline().rstrip())#函数会去除前后空格后的字符串 12
        image_paths=[]
        for i in range(image_nums):
            image_path=file.readline().rstrip()
            image_paths.append(image_path)
        image_mask_path=file.readline().rstrip()#最后一行是掩模路径
        #开始加载图片
    images=[]
    for image_path_ in image_paths:
        image=image_open(image_path_)
        images.append(image)

    mask_image=image_open(image_mask_path)
    mask_image=cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    print("加载数据成功")

    return images,mask_image



