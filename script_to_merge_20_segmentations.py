import os
import numpy as np
import cv2
import transformations
import pandas as pd
import random
import transformations as t
import matplotlib.pyplot as plt
from PIL import Image

#%%
or_directory = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/DATASET/train/train/manual/"
or_name = "1001001_16.png"
ric_directory = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/DATASET/train/train/perturbated_segmentations/"
ric_name = "ric_1001001_16/"

or_seg = cv2.imread(or_directory+or_name, cv2.COLOR_BGR2GRAY)

list_of_images = ric_directory+ric_name
final_image = np.zeros((416,416,20))
n=0
for img_name in os.listdir(list_of_images):
    image_n = cv2.imread(list_of_images+img_name, cv2.IMREAD_UNCHANGED)
    final_image[:,:,n]=image_n
    n+=1

definitive_final_image = np.zeros((416,416))
for i in range(416):
    for j in range(416):
        definitive_final_image[i][j] = (final_image[i][j][:]).sum()
        
massimo = definitive_final_image.max()

definitive_final_image /= massimo

diff = definitive_final_image-np.float32(or_seg)/255

plt.imshow(diff)
print(np.count_nonzero(diff))
