import numpy as np
from skimage import transform
from scipy import ndimage
import os

# data = np.load('F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176\Image_ROI/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_45_211_77.npy', allow_pickle=True)
# print(data.shape)
# data_resize = transform.resize(data, (48,48,48))
# print(data_resize.shape)

def min_max_normalize(image):
    # 数值范围为-1024到2000左右，对400以上的不感兴趣，所以在-1400到400之间归一化
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image[image > MAX_BOUND] = MAX_BOUND
    image[image < MIN_BOUND] = MIN_BOUND
    image = (image-image[:,:,:].mean())/image[:,:,:].std()
    return image

'''class Resize3D:
    def __init__(self, target_size=[48, 48, 48], model='constant', order=1):
        self.model = model
        self.order = order
        self.target_size = target_size
    def __call__(self, im):
        desired_depth = self.target_size[2]  # 深度
        desired_width = self.target_size[1]
        desired_height = self.target_size[0]

        current_depth = im.shape[2]  # 深度
        current_width = im.shape[1]
        current_height = im.shape[0]

        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height

        im = ndimage.zoom(im, (height_factor, width_factor, depth_factor), order=self.order, mode=self.model)
        return im'''

image_path = 'F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176\Image_ROI'    # 文件夹路径
out_file_dir = 'F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176\Image_resize'
image_file = os.listdir(image_path)   # 遍历该文件夹下所有的文件

target_size = [48, 48, 48]
image_shape = [33,33,18]

desired_depth = target_size[2]  # 深度
desired_width = target_size[1]
desired_height = target_size[0]

current_depth = image_shape[2]  # 深度
current_width = image_shape[1]
current_height = image_shape[0]

depth = current_depth / desired_depth
width = current_width / desired_width
height = current_height / desired_height
depth_factor = 1 / depth
width_factor = 1 / width
height_factor = 1 / height

for index in range(len(image_file)):
    image = np.load(image_path + '/' + image_file[index])
    image_norm = normalize(image)
    image_resize = transform.resize(image_norm,(48, 48, 48))
    # image_resize = ndimage.zoom(image_norm, (height_factor, width_factor, depth_factor), order=1, mode='constant')
    # print(image_resize.shape)
    np.save(out_file_dir + '/' + image_file[index], image_resize)
