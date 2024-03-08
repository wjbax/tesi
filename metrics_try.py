import os
import numpy as np
import pandas as pd
import metrics_v2 as m
import matplotlib.pyplot as plt
# import cv2
from tqdm import tqdm
import random
import PIL.Image as Img

#%% PATH INITIALIZATION
SOFTMAX_DIM = [416,416]
GT_seg_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/DATASET/test/manual/"
softmax_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/softmax/"
seg_MC_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS/test/mask/"

print(os.listdir(GT_seg_dir))
#%%
# DATASET = np.zeros([50,5])
# i = 0
# for GT_seg_name in tqdm(os.listdir(GT_seg_dir)):
#     softmax_path = softmax_dir + GT_seg_name + "/"
#     seg_MC_path = seg_MC_dir + GT_seg_name
#     MC_softmax_list = os.listdir(softmax_path)

GT_seg_name = '1004289_35.png'
softmax_path = softmax_dir + GT_seg_name + "/"
seg_MC_path = seg_MC_dir + GT_seg_name
MC_softmax_list = os.listdir(softmax_path)

seg_GT = Img.open(GT_seg_dir + GT_seg_name)
seg_MC = Img.open(seg_MC_path)

#%%
seg_GT.show()
seg_MC.show()

#%%
X = np.array(seg_GT)/255
Y = np.array(seg_MC)/255
# X = np.zeros((50,50))
# Y = np.copy(X)
#%%
print(m.dice(X,Y))

#%%
Z = X+Y
halfnum = np.count_nonzero(X*Y>0) 
denom = np.sum(Z)
dice = 2*halfnum/denom
print(dice)

#%%
A = np.array([1,0,1,1,0,0,1,0])
B = np.array([1,0,1,1,1,0,1,0])

C = A+B
print(C)

halfnum = np.count_nonzero(A*B>0) 
denom = np.sum(C)
dice = 2*halfnum/denom
print(dice)

#%%
mask1 = X
mask2 = Y
intersect = np.sum(mask1*mask2)
fsum = np.sum(mask1)
ssum = np.sum(mask2)
dice2 = (2 * intersect ) / (fsum + ssum)
print(dice2)



