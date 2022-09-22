import pandas as pd
import torch
import os
from model import *
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def trainmutilresnet3d():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data_Malignancy\\traindata.csv')
    maskdatasource = csvdata.iloc[:, 1].values
    imagedatasource = csvdata.iloc[:, 0].values
    csvdataaug = pd.read_csv('dataprocess\\data_Malignancy\\trainaugdata.csv')
    maskdataaug = csvdataaug.iloc[:, 1].values
    imagedataaug = csvdataaug.iloc[:, 0].values
    imagedata = np.concatenate((imagedatasource, imagedataaug), axis=0)
    maskdata = np.concatenate((maskdatasource, maskdataaug), axis=0)
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess/data_Malignancy/validata.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    resnet3d = MutilResNet3dModel_metric(image_depth=48, image_height=48, image_width=48, image_channel=1, numclass=2,
                              batch_size=16, loss_name='MutilCrossEntropyLoss', inference=True, model_path='log_Malignancy/MutilResNet3dModel/dice/VGG3d/MutilResNet3d.pth')
    resnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log_Malignancy/MutilResNet3dModel/dice/VGG3d',
                        epochs=200, lr=1e-3)


if __name__ == '__main__':
    trainmutilresnet3d()
