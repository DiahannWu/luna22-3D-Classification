from __future__ import print_function, division

import os
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import file_name_path, resize_image_itkwithsize, normalize, ConvertitkTrunctedValue

image_dir = "Image_ROI"
mask_dir = "Mask"
outimage_dir = 'Image_resize'

# def preparesampling3dtraindata(datapath, trainImage, trainMask, shape=(96, 96, 96)):
#     newSize = shape
#     dataImagepath = datapath + "/" + image_dir
#     dataMaskpath = datapath + "/" + mask_dir
#     all_files = file_name_path(dataImagepath, False, True)
#     for subsetindex in range(len(all_files)):
#         mask_name = all_files[subsetindex]
#         mask_gt_file = dataMaskpath + "/" + mask_name
#         masksegsitk = sitk.ReadImage(mask_gt_file)
#         image_name = all_files[subsetindex]
#         image_gt_file = dataImagepath + "/" + image_name
#         imagesitk = sitk.ReadImage(image_gt_file)
#
#         _, resizeimage = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(),
#                                                   sitk.sitkLinear)
#         _, resizemask = resize_image_itkwithsize(masksegsitk, newSize, masksegsitk.GetSize(),
#                                                  sitk.sitkNearestNeighbor)
#         # sitk.WriteImage(resizeimage, 'resizeimage.nii.gz')
#         # sitk.WriteImage(resizemask, 'resizemask.nii.gz')
#         resizemaskarray = sitk.GetArrayFromImage(resizemask)
#         resizeimagearray = sitk.GetArrayFromImage(resizeimage)
#         resizeimagearray = normalize(resizeimagearray)
#         # step 3 get subimages and submasks
#         if not os.path.exists(trainImage):
#             os.makedirs(trainImage)
#         if not os.path.exists(trainMask):
#             os.makedirs(trainMask)
#         filepath1 = trainImage + "\\" + str(subsetindex) + ".npy"
#         filepath = trainMask + "\\" + str(subsetindex) + ".npy"
#         np.save(filepath1, resizeimagearray)
#         np.save(filepath, resizemaskarray)


# def preparetraindata():
#     """
#     :return:
#     """
#     src_train_path = r"processstage\train"
#     source_process_path = r"trainstage\train"
#     outputimagepath = source_process_path + "/" + image_dir
#     outputlabelpath = source_process_path + "/" + mask_dir
#     preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (112, 112, 128))


def preparesampling3dtraindata(datapath, trainImage, shape=(96, 96, 96)):
    newSize = shape
    dataImagepath = datapath + "/" + image_dir
    # dataMaskpath = datapath + "/" + mask_dir
    all_files = file_name_path(dataImagepath, False, True)
    for subsetindex in range(len(all_files)):
        # mask_name = all_files[subsetindex]
        # mask_gt_file = dataMaskpath + "/" + mask_name
        # masksegsitk = sitk.ReadImage(mask_gt_file)
        image_name = all_files[subsetindex]
        image_gt_file = dataImagepath + "/" + image_name
        imagesitk = sitk.ReadImage(image_gt_file)

        _, resizeimage = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(),
                                                  sitk.sitkLinear)
        # resizeimage = ConvertitkTrunctedValue(resizeimage,400,-1000,'meanstd') ###
        # _, resizemask = resize_image_itkwithsize(masksegsitk, newSize, masksegsitk.GetSize(),
        #                                          sitk.sitkNearestNeighbor)
        # sitk.WriteImage(resizeimage, 'resizeimage.nii.gz')
        # sitk.WriteImage(resizemask, 'resizemask.nii.gz')
        # resizemaskarray = sitk.GetArrayFromImage(resizemask)
        resizeimagearray = sitk.GetArrayFromImage(resizeimage)
        # resizeimagearray = normalize(resizeimagearray)
        # step 3 get subimages and submasks
        if not os.path.exists(trainImage):
            os.makedirs(trainImage)
        # if not os.path.exists(trainMask):
        #     os.makedirs(trainMask)
        filepath1 = trainImage + "\\" + image_name[:-12] + ".npy"
        # filepath = trainMask + "\\" + str(subsetindex) + ".npy"
        np.save(filepath1, resizeimagearray)
        # np.save(filepath, resizemaskarray)

def preparetraindata():
    """
    :return:
    """
    src_train_path = r"F:\myfile\detection\PytorchDeepLearing\processeddata"
    source_process_path = r"F:\myfile\detection\PytorchDeepLearing\processeddata"
    outputimagepath = source_process_path + "/" + outimage_dir
    # outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, (48, 48, 48))



def preparevalidationdata():
    """
    :return:
    """
    src_train_path = r"processstage\validation"
    source_process_path = r"trainstage\validation"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (112, 112, 128))


if __name__ == "__main__":
    preparetraindata()
    # preparevalidationdata()
