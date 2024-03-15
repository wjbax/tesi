# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:07:00 2024

@author: willy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
def dice(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()
    # TN_mask = np.multiply(~mask_automatic,~mask_manual); TN = TN_mask.sum()
    
    if TP==0 and FN==0 and FP==0:
        # jaccard_ind = np.nan
        dice_ind = np.nan
    else:
        # jaccard_ind = TP/(TP+FP+FN)
        dice_ind = 2*TP/(2*TP+FP+FN)
    return dice_ind
#%%
path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
metrics_name = "metrics_v6.csv"
dice_mat_name = "dice_mat_tracker.npy"
seg_tracker_name = "seg_matrix_tracker.npy"
softmax_tracker_name = "softmax_matrix_tracker.npy"

#%%
softmax_matrix_tracker = np.load(path+softmax_tracker_name)
seg_matrix_tracker = np.load(path+seg_tracker_name)
dice_mat_tracker = np.load(path+dice_mat_name)
metrics_df = pd.read_csv(path+metrics_name)

#%%
num_img = 6
num_MC = 3

#%%
actual_softmax = softmax_matrix_tracker[:,:,1,num_MC,num_img]
actual_seg = seg_matrix_tracker[:,:,num_MC,num_img]
actual_dice_mat = dice_mat_tracker[:,:,num_img]
image_name = metrics_df['image_tracker'][num_img-1]

#%%
plt.imshow(actual_softmax)
plt.imshow(actual_seg)
plt.imshow(actual_dice_mat)
print(image_name)

#%%
dice_mat_tracker_bool = dice_mat_tracker.astype(bool)

dice_mat_temp = -np.ones((20,20))
dice_array_GT = []
for i in range(20):
    for j in range(i+1,20):
        dice_mat_temp[i,j] = dice(seg_matrix_tracker[:,:,i,num_img],seg_matrix_tracker[:,:,j,num_img])
dice_mat_temp[dice_mat_temp<0] = np.nan

plt.imshow(dice_mat_temp)

plt.imshow(seg_matrix_tracker[:,:,i,num_img])
