from scipy import ndimage
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import os

MIN_BOUND = -1000.0
MAX_BOUND = -500


def norm_img_func(image):  # 归一化像素值到（0，1）之间，且将溢出值取边界值
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


class Resize3D:
    def __init__(self, target_size=[48, 48, 48], model='constant', order=1):
        self.model = model
        self.order = order

        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 3:
                raise ValueError(
                    '`target_size` should include 3 elements, but it is {}'.
                        format(target_size))

        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                    .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, label=None):
        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if im.ndim == 3:
            desired_depth = depth = self.target_size[2]  # 深度
            desired_width = width = self.target_size[1]
            desired_height = height = self.target_size[0]

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
            if label is not None:
                label = ndimage.zoom(label, (depth_factor, width_factor, height_factor), order=0, mode='nearest',
                                     cval=0.0)

        else:
            # 通道方面错误，需要修改
            channels = [
                ndimage.zoom(im[c], (depth_factor, width_factor, height_factor), order=self.order, mode=self.model) for
                c in range(im.shape[0])]
            im = np.stack(channels, axis=0)
            if label is not None:
                channels = [
                    ndimage.zoom(label[c], label, (depth_factor, width_factor, height_factor), order=0, mode='nearest',
                                 cval=0.0) for c in range(label.shape[0])]
                label = np.stack(channels, axis=0)

        if label is None:
            return im
        else:
            return (im, label)


if __name__ == '__main__':
    image_path = "D:/airway_datasets/train_image_nii/CASE01.nii.gz"
    filename = os.path.basename(image_path).split('.')[0]
    print(filename)  # CASE01

    func = nib.load(image_path)
    img2 = np.array(func.get_fdata())
    print(f'原始图像的shape: {img2.shape}')

    h, w, d = img2.shape
    maxiter = np.max([h, w, d]) // 128
    resolution = (maxiter + 1) * 128
    print(resolution)
    # print(f'd维还剩{d-maxiter*128}')
    # img0 = img2[0:128,0:128,0:128]
    # print(img0.shape)

    # plt.imshow(img2[:,:,0],cmap='gray')
    # plt.show()

    # 1. 先将三维图像resize到resolution
    img_resize = Resize3D([resolution, resolution, resolution])(img2)
    print(f'resize之后的shape: ', img_resize.shape)

    # # 2. 保存看一下
    # new_image = nib.Nifti1Image(img_resize,func.affine)
    # nib.save(new_image,'out_01.nii')

    # 2. 对resize之后的图像进行切割
    cnt = 0
    for i in range(maxiter + 1):
        for j in range(maxiter + 1):
            for k in range(maxiter + 1):
                # 0-128,0-128,0-128
                # 128-256
                # ...
                # 512-640
                imgt = img_resize[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128, k * 128:(k + 1) * 128]
                new_image = nib.Nifti1Image(imgt, func.affine)
                nib.save(new_image, f'case01/{filename}_{cnt:03}.nii')
                print(f'save {cnt:03} done.')
                cnt += 1
