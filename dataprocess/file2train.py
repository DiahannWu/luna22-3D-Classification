import os
import numpy as np

def texture_classes(data):
    data.sort()
    half = len(data) // 2
    mid_value = (data[half] + data[~half]) / 2
    value = mid_value
    if (int(mid_value) <= 4) & (int(mid_value) >= 2):
        value = 2
    elif mid_value == 5:
        value = 3
    return value

def Maliganacy_classes(data):
    data.sort()
    half = len(data) // 2
    mid_value = (data[half] + data[~half]) / 2
    if mid_value <= 2:
        value = 1
    else:
        value = 2
    return value

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def save_file2csv(file_dir, file_name, val_file_name, percent):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    val_out = open(val_file_name, 'w')
    image = "Image_resize"
    mask = "Mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask

    # file_paths = file_name_path(file_image_dir, dir=False, file=True)
    labels_information = np.load(file_mask_dir + '/LIDC-IDRI_1176.npy', allow_pickle=True)
    out.writelines("Image,Mask" + "\n")
    len_train = int(len(labels_information)*percent)
    for index in range(len_train):
        # out_file_image_path = file_image_dir + "/" + file_paths[index]
        # labels_information = np.load(file_mask_dir+'/LIDC-IDRI_1176.npy', allow_pickle=True)
        label_list = labels_information[index]['Texture']
        label = int(round(texture_classes(label_list)))-1  # 四舍五入取整数
        out_file_image_path = file_image_dir + "/" + labels_information[index]['Filename'][:-12]+'.npy'
        out.writelines(out_file_image_path + "," + str(label) + "\n")

    val_out.writelines("Image,Mask" + "\n")
    for index in range(len_train, len(labels_information)):
        # out_file_image_path = file_image_dir + "/" + file_paths[index]
        # labels_information = np.load(file_mask_dir+'/LIDC-IDRI_1176.npy', allow_pickle=True)
        label_list = labels_information[index]['Texture']
        label = int(round(texture_classes(label_list)))-1  # 四舍五入取整数
        out_file_image_path = file_image_dir + "/" + labels_information[index]['Filename'][:-12]+'.npy'
        val_out.writelines(out_file_image_path + "," + str(label) + "\n")


def save_file2csv1(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "Image"
    mask = "Mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask

    # file_paths = file_name_path(file_image_dir, dir=False, file=True)
    labels_information = np.load(file_mask_dir + '/LIDC-IDRI_1176.npy', allow_pickle=True)
    out.writelines("Image,Mask" + "\n")
    len_train = len(labels_information)
    for index in range(len_train):
        # out_file_image_path = file_image_dir + "/" + file_paths[index]
        label_list = labels_information[index]['Texture']
        label = int(round(texture_classes(label_list)))-1  # 四舍五入取整数
        # Malignancy
        # label_list = labels_information[index]['Malignancy']
        # label = int(round(Maliganacy_classes(label_list))) - 1  # 四舍五入取整数
        out_file_image_path = file_image_dir + "/" + labels_information[index]['Filename'][:-12]+'.npy'
        out.writelines(out_file_image_path + "," + str(label) + "\n")


def save_file2csv2(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "Image"
    mask = "Mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask

    file_paths = file_name_path(file_image_dir, dir=False, file=True)
    out.writelines("Image,Mask" + "\n")
    len_train = len(file_paths)
    for index in range(len_train):
        out_file_image_path = file_image_dir + "/" + file_paths[index]
        out_file_mask_path = file_mask_dir + "/" + file_paths[index]
        label = int(np.load(out_file_mask_path, allow_pickle=True))
        out.writelines(out_file_image_path + "," + str(label) + "\n")


def save_file2csv_Malignancy(file_dir, file_name, val_file_name, percent):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    val_out = open(val_file_name, 'w')
    image = "Image_resize"
    mask = "Mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask

    # file_paths = file_name_path(file_image_dir, dir=False, file=True)
    labels_information = np.load(file_mask_dir + '/LIDC-IDRI_1176.npy', allow_pickle=True)
    out.writelines("Image,Mask" + "\n")
    len_train = int(len(labels_information)*percent)
    for index in range(len_train):
        # out_file_image_path = file_image_dir + "/" + file_paths[index]
        # labels_information = np.load(file_mask_dir+'/LIDC-IDRI_1176.npy', allow_pickle=True)
        label_list = labels_information[index]['Malignancy']
        label = int(round(Maliganacy_classes(label_list)))-1  # 四舍五入取整数
        out_file_image_path = file_image_dir + "/" + labels_information[index]['Filename'][:-12]+'.npy'
        out.writelines(out_file_image_path + "," + str(label) + "\n")

    val_out.writelines("Image,Mask" + "\n")
    for index in range(len_train, len(labels_information)):
        # out_file_image_path = file_image_dir + "/" + file_paths[index]
        # labels_information = np.load(file_mask_dir+'/LIDC-IDRI_1176.npy', allow_pickle=True)
        label_list = labels_information[index]['Malignancy']
        label = int(round(Maliganacy_classes(label_list)))-1  # 四舍五入取整数
        out_file_image_path = file_image_dir + "/" + labels_information[index]['Filename'][:-12]+'.npy'
        val_out.writelines(out_file_image_path + "," + str(label) + "\n")


if __name__ == '__main__':
    # save_file2csv(r'D:\challenge\data\KiPA2022\trainstage\train', 'data/traindata.csv')
    # save_file2csv(r'D:\challenge\data\KiPA2022\trainstage\validation', 'data/validata.csv')
    # save_file2csv(r'D:\challenge\data\KiPA2022\trainstage\augtrain', 'data/trainaugdata.csv')
    # save_file2csv(r'F:/myfile/detection/PytorchDeepLearing/originaldata/LIDC-IDRI_1176', 'data_250/traindata.csv', 'data_250/validata.csv', percent=0.8)
    # save_file2csv1(r'F:/myfile/detection/PytorchDeepLearing/originaldata/LIDC-IDRI_1176', 'data/alldata.csv')
    # save_file2csv2(r'F:/myfile/detection/PytorchDeepLearing/originaldata/LIDC-IDRI_1176/augtrain_resize', 'data_resize/trainaugdata.csv')
    # save_file2csv_Malignancy(r'F:\myfile\detection\PytorchDeepLearing\processeddata', 'data_Malignancy/traindata.csv', 'data_Malignancy/validata.csv', percent=0.8)
    # save_file2csv1(r'F:\myfile\detection\PytorchDeepLearing\processeddata', 'data_Malignancy/alldata.csv')
    save_file2csv2(r'F:\myfile\detection\PytorchDeepLearing\processeddata/augtrain_Malignancy', 'data_Malignancy/trainaugdata.csv')
