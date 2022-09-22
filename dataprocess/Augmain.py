from dataprocess.Augmentation.ImageAugmentation import DataAug3D

if __name__ == '__main__':
    aug = DataAug3D(rotation=10, width_shift=0.01, height_shift=0.01, depth_shift=0, zoom_range=0,
                    vertical_flip=True, horizontal_flip=True)
    label = 0
    if label==0:
        multi=24  # 10
    else:
        multi=4
    aug.DataAugmentation('data_Malignancy/traindata{}.csv'.format(label), multi,
                         aug_path='F:\myfile\detection\PytorchDeepLearing\processeddata/augtrain_Malignancy/', aug_label=label)
