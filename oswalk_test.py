import numpy as np
import torch
from matplotlib import pyplot as plt

## 加载npy文件
# labelinformation = np.load('F:\myfile\detection\PytorchDeepLearing\originaldata/LIDC-IDRI_1176.npy', allow_pickle=True)
# print(labelinformation)

## ------------------nii图像可视化-----------------

## 2维显示
import skimage.io as io
import SimpleITK as sitk

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

#显示一个系列图
# def show_img(data):
#     for i in range(data.shape[0]):
#         io.imshow(data[i,:,:], cmap='gray')
#         print(i)
#         io.show()


#单张显示
def show_img(ori_img):
    io.imshow(ori_img[60], cmap='gray')
    io.show()

path = 'originaldata/LIDC-IDRI_1176/LIDC-IDRI/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_45_211_77_0000.nii.gz'
data = read_img(path)
print(data.shape)
show_img(data)


# '''from mayavi import mlab
# import SimpleITK as sitk
# from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
#
# img_path = 'originaldata/LIDC-IDRI_1176/LIDC-IDRI/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_405_154_117_0000.nii.gz'
#
# # image_path = 'xxx'
# img = sitk.ReadImage(img_path)
# img_arr = sitk.GetArrayFromImage(img)
# C, H, W = img_arr.shape
# print("img shape =", img_arr.shape)  # C, H, W
# img_arr = img_arr[:C, :H, :W]
#
# ## mayavi
# # 表面绘制
# # mlab.contour3d(img_arr)
#
# # 体绘制
# vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(img_arr), name='3d-ultrasound')
#
# ctf = ColorTransferFunction()  # 该函数决定体绘制的颜色、灰度等
# vol._ctf = ctf
# vol._volume_property.set_color(ctf)  # 进行更改，体绘制的colormap及color
# vol.update_ctf = True
#
# otf = PiecewiseFunction()
# otf.add_point(20, 0.2)
# vol._otf = otf
# vol._volume_property.set_scalar_opacity(otf)
#
# # mlab.volume_slice(img_arr, colormap='gray',
# #                   plane_orientation='z_axes', slice_index=W//2)          # 设定z轴切面
#
# mlab.show()
# '''
# ## ------------------nii图像可视化-----------------

# print(image.shape)
# image = np.transpose(image, (2,0,1))
# print(image[0].shape)
# plt.imshow(image[0], cmap='gray')  # 灰度图展示
#
# plt.show()


# import os
#
# def file_name_path(file_dir, dir=True, file=False):
#     """
#     get root path,sub_dirs,all_sub_files
#     :param file_dir:
#     :return: dir or file
#     """
#     for root, dirs, files in os.walk(file_dir):
#         print('hhhh')
#         if len(dirs) and dir:
#             print("sub_dirs:", dirs)
#             return dirs
#         if len(files) and file:
#             print("files:", files)
#             return files
#
# a= 'originaldata\LIDC-IDRI_1176'
# b= 'Image'
# path = a + '/' + b
#
# filepaths = file_name_path(path, dir=False,file=True)
# print(filepaths)
#
#
# # for root, dirs, files in os.walk("originaldata\LIDC-IDRI_1176/Image", topdown=False):
# #     print(len(files))
# #     for name in files:
# #         print(os.path.join(root, name))
# #     # for name in dirs:
# #     #     print(os.path.join(root, name))
