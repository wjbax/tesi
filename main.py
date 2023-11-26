import os
import numpy as np
import cv2
import transformations
import pandas as pd
import random

#%% Dataset path and directories
DATASET_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/"
directories = [d for d in os.listdir(DATASET_path) if os.path.isdir(os.path.join(DATASET_path, d))]
## IMPORTANT: I changed the structure of the dataset splitting the Prostate HE (glands, tumor)
## directory into Prostate HE glands and Prostate HE tumor directories, also changing the name
## of the DATASET directory present in them from DATASET_glands or DATASET_tumor to DATASET to
## match the structure of the other directories: we have 5 directories now

#%%
image_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/DATASET/train/train/image/1001051_19.png"
seg_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/DATASET/train/train/manual/1001051_19.png"
softmax_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/k-net+swin/TEST_2classes/RESULTS/train/softmax/1001051_19.npz"

list_of_transformations = {
    "Rotation" :                transformations.rotation,
    "Vertical mirroring" :      transformations.vert_mirr,
    "Horizontal mirroring" :    transformations.hor_mirr,
    "Scaling" :                 transformations.scaling,
    "Gaussian blurring" :       transformations.gaussblur,
    "HSV perturbations" :       transformations.hsv_pert
}
#%%
image = cv2.imread(image_path)
image = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB)
image = image.astype(np.float32)

seg = cv2.imread(seg_path)
seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
seg = np.float32(seg)

b = np.load(softmax_path)['softmax']
softmax = np.float32(b[:,:,1])

original_shape = np.shape(image)

#%%
for directory in directories:
  transformations_track_table = pd.DataFrame(columns=['IMG_Name', 'Transformations', 'Factor/std/angle'])
  img_directory = DATASET_path + directory + '/DATASET/train/train/image/'
  img_saving_directory = DATASET_path + directory + '/DATASET/train/train/artificial_images/'
  segm_directory = DATASET_path + directory + "/DATASET/train/train/manual/"
  if not os.path.exists(img_saving_directory):
    os.makedirs(img_saving_directory)
  img_list = os.listdir(img_directory)
  for img_name in img_list:
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    seg = cv2.imread(seg_path)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    seg = np.float32(seg)
    b = np.load(softmax_path)['softmax']
    softmax = np.float32(b[:,:,1])
    original_shape = np.shape(image)
    
    selected_functions = random.sample(list(list_of_transformations.keys()), random.randint(1,4))

    for i in selected_functions:
        le cose