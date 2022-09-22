import nibabel as nib
import os
import numpy as np

img_path = '../originaldata\LIDC-IDRI_1176\LIDC-IDRI/'
saveimg_path = '../originaldata\LIDC-IDRI_1176\LIDC-IDRI-npy/'

if not os.path.exists(saveimg_path):
    os.makedirs(saveimg_path)

img_names = os.listdir(img_path)


for img_name in img_names:
    print(img_name)
    img = nib.load(img_path + img_name).get_data() #载入
    img = np.array(img)
    np.save(saveimg_path + str(img_name)[:-12] + '.npy', img) #保存

