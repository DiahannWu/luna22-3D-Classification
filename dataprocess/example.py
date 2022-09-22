import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
# filename = 'F:\myfile\detection\PytorchDeepLearing\processstage\\train\Image/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_45_211_77_0000.nii.gz'
# img = nib.load(filename)
# print(img)

# csvXdata = pd.read_csv('data/traindata.csv')
# data = csvXdata.iloc[:, :].values
# print(data[0][0])

# for i in range(0,5):
#     data = np.load('F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176\\augtrain\Image/{}_10.npy'.format(i))
#     print(data.shape)


data_information = np.load('F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176\Mask/LIDC-IDRI_1176.npy', allow_pickle=True)
print(data_information[0])
# # print(data_information[0]['VoxelCoordX'])
# print(data_information[1])

# diameters = []
# for index in range(len(data_information)):
#     if index == 361:
#         print(data_information[index]['Diameter'])
#         print('============')
#     max_dimeter = np.mean(data_information[index]['Diameter'])
#     # print(max_dimeter)
#     diameters.append(max_dimeter)
# print('=========================')
# print(max(diameters))
# print(diameters.index(max(diameters)))

'''#显示多张图片
def show_images(data):
    center = int((data.shape[2])/2)
    plt.figure()  # figsize=(10,10)
    for i, slice in enumerate([0, center-2, center, center+2, data.shape[2]-1]):
        plt.subplot(1,5,i+1)
        plt.title('slice{}'.format(slice), y=-0.3)
        plt.axis('off')
        plt.imshow(data[:,:,slice], cmap='gray')
        # plt.imshow(data[slice, :, :], cmap='gray')
        # plt.imshow(data[:, slice, :], cmap='gray')
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig('F:\myfile\detection\PytorchDeepLearing/CT_figures/origin_dim2_5.png', bbox_inches='tight')
    plt.show()

# 显示单张图片
def show_single_image(data):
    for i in range(data.shape[2]):
        plt.axis('off')
        plt.imshow(data[:,:,i], cmap='gray')
        plt.savefig('F:\myfile\detection\PytorchDeepLearing/CT_figures/images/{}.png'.format(i), bbox_inches='tight')
        # plt.show()


# data1 = np.load('F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176/LIDC-IDRI-npy/1.3.6.1.4.1.14519.5.2.1.6279.6001.119806527488108718706404165837_129_140_179.npy', allow_pickle=True)
# data1 = np.load('F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176/LIDC-IDRI-npy/1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800_333_328_113.npy', allow_pickle=True)
# data2 = np.load('F:\myfile\detection\PytorchDeepLearing\originaldata\LIDC-IDRI_1176\\augtrain_ROI/Image/00_1.npy', allow_pickle=True)
imagesitk = sitk.ReadImage('F:\myfile\detection\PytorchDeepLearing\processeddata\Image_ROI/1.3.6.1.4.1.14519.5.2.1.6279.6001.119806527488108718706404165837_129_140_179_0000.nii.gz')
# imagesitk = sitk.ReadImage('F:\myfile\detection\PytorchDeepLearing\processeddata\Image_ROI/1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800_333_328_113_0000.nii.gz')
data1 = sitk.GetArrayFromImage(imagesitk)
print(data1.shape)
show_images(data1)'''




'''
# 图像3维显示
data1 = data1.transpose((2, 1, 0))
print(data1.shape)

C, H, W = data1.shape
print("img shape =", data1.shape)  # C, H, W
img_arr = data1[:C, :H, :W//2]
# print(data2.shape)
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
# 表面绘制
# mlab.contour3d(data1)

# 体绘制
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data1), name='3d-ultrasound')

ctf = ColorTransferFunction()  # 该函数决定体绘制的颜色、灰度等
vol._ctf = ctf
vol._volume_property.set_color(ctf)  # 进行更改，体绘制的colormap及color
vol.update_ctf = True

otf = PiecewiseFunction()
otf.add_point(20, 0.2)
vol._otf = otf
vol._volume_property.set_scalar_opacity(otf)
# mlab.volume_slice(data1, colormap='gray',
#                   plane_orientation='z_axes', slice_index=W//2)
mlab.show()'''

# data_roi = data[47:80, 47:80, 22:40]
# print(data_roi.shape)


