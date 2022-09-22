import os
import numpy as np
import SimpleITK as sitk

image_dir = 'LIDC-IDRI'
labels_dir = 'LIDC-IDRI_1176.npy'


def setMetaMessage(target, origin):

    target.SetDirection(origin.GetDirection())
    target.SetOrigin(origin.GetOrigin())
    target.SetSpacing(origin.GetSpacing())
    return target


def get_roi_data(datapath, outputImage):  # , shape=(96, 96, 96)
    # newSize = shape
    dataImagepath = datapath + "/" + image_dir
    labelpath = datapath + '/' + labels_dir
    labels_information = np.load(labelpath, allow_pickle=True)
    for subsetindex in range(len(labels_information)):
        # mask_name = all_files[subsetindex]
        # mask_gt_file = dataMaskpath + "/" + mask_name
        # masksegsitk = sitk.ReadImage(mask_gt_file)
        image_name = labels_information[subsetindex]['Filename']
        image_gt_file = dataImagepath + "/" + image_name
        imagesitk = sitk.ReadImage(image_gt_file)
        spacing = imagesitk.GetSpacing()

        image = sitk.GetArrayFromImage(imagesitk)
        image = image.transpose(2, 1, 0)
        # voxelcoordX = labels_information[subsetindex]['VoxelCoordX']
        # voxelcoordY = labels_information[subsetindex]['VoxelCoordY']
        # voxelcoordZ = labels_information[subsetindex]['VoxelCoordZ']
        shape = image.shape
        offsetx = max(labels_information[subsetindex]['Diameter'])/spacing[0]
        offsety = max(labels_information[subsetindex]['Diameter'])/spacing[1]
        offsetz = max(labels_information[subsetindex]['Diameter'])/spacing[2]
        image = image[int((shape[0]-offsetx)/2):int((shape[0]+offsetx)/2),
                int((shape[1]-offsety)/2):int((shape[1]+offsety)/2),
                int((shape[2]-offsetz)/2):int((shape[2]+offsetz)/2)]
        # _, resizemask = resize_image_itkwithsize(masksegsitk, newSize, masksegsitk.GetSize(),
        #                                          sitk.sitkNearestNeighbor)
        # sitk.WriteImage(resizeimage, 'resizeimage.nii.gz')
        # sitk.WriteImage(resizemask, 'resizemask.nii.gz')
        # resizemaskarray = sitk.GetArrayFromImage(resizemask)
        image_roi = sitk.GetImageFromArray(image)
        image_roi = setMetaMessage(image_roi, imagesitk)

        # resizeimagearray = normalize(resizeimagearray)
        # step 3 get subimages and submasks
        if not os.path.exists(outputImage):
            os.makedirs(outputImage, exist_ok=True)
        filepath = outputImage + '/' + image_name
        sitk.WriteImage(image_roi, filepath)
        # if not os.path.exists(trainMask):
        #     os.makedirs(trainMask)
        # filepath1 = trainImage + "\\" + str(subsetindex) + ".npy"
        # filepath = trainMask + "\\" + str(subsetindex) + ".npy"
        # np.save(filepath1, resizeimagearray)
        # np.save(filepath, resizemaskarray)
if __name__ == '__main__':
    src_file_path = 'F:/myfile/detection/PytorchDeepLearing/originaldata/LIDC-IDRI_1176'
    out_iamge_path = 'F:/myfile/detection/PytorchDeepLearing/processeddata/Image_ROI'
    get_roi_data(src_file_path, out_iamge_path)



# file_image_dir = 'F:/myfile/detection/PytorchDeepLearing/originaldata/LIDC-IDRI_1176/Image'
# out_file_dir = 'F:/myfile/detection/PytorchDeepLearing/originaldata/LIDC-IDRI_1176/Image_ROI'
# if not os.path.exists(out_file_dir):
#     os.makedirs(out_file_dir, exist_ok=True)
# # file_paths = file_name_path(file_image_dir, dir=False, file=True)
# file_paths = os.listdir(file_image_dir)
# # print(file_paths)
#
# for file_name in file_paths:
#     # print(file_name)
#     # print(file_paths[index])
#     data = np.load(file_image_dir + "/" + file_name, allow_pickle=True)
#     data_roi = data[47:80, 47:80, 22:40]
#     np.save(out_file_dir + '/' + file_name, data_roi)
