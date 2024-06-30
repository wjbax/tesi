# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:44:29 2024

@author: willy
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import measure
import copy
from tqdm import tqdm
import wjb_functions
import pandas as pd

#%%

diff_type = "PERT"
diff_type_name = "_perturbation/"
dataset = "Liver HE steatosis/"
subset = "test"
image = "1004289_35.png"
N = 20
radius = 5

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V10/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"
GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image
SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
diff_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
diff_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
softmax_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"

#%%

GT_mask_2c = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
SI_mask_2c = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
mask_union_2c = wjb_functions.mask_union_gen(diff_path_2c)
mask_union_3c_int = wjb_functions.mask_union_gen(diff_path_3c)
softmax_matrix_2c = wjb_functions.softmax_matrix_gen(softmax_path_2c, np.shape(GT_mask_2c)[:2], 2, N)
softmax_matrix_3c = wjb_functions.softmax_matrix_gen(softmax_path_3c, np.shape(GT_mask_2c)[:2], 3, N)
unc_map_2c = wjb_functions.binary_entropy_map(softmax_matrix_2c[:,:,1,:])
unc_map_3c = wjb_functions.binary_entropy_map(softmax_matrix_3c[:,:,1,:])

label_image = measure.label(mask_union_3c_int)
n_objects = label_image.max()
mask_matrix = np.zeros([label_image.shape[0],label_image.shape[1],n_objects])
tau_array_2c = []
tau_array_3c = []
for i in range(n_objects):
    current_mask = np.copy(label_image)
    current_mask[current_mask!=i+1] = 0
    max_mask = np.max(current_mask)
    if max_mask == 0: max_mask = 1
    current_mask = current_mask/max_mask
    mask_matrix[:,:,i] = current_mask
    unc_map_temp_2c = np.multiply(current_mask,unc_map_2c)
    unc_map_temp_3c = np.multiply(current_mask,unc_map_3c)
    tau_i_2c = np.nanmean(unc_map_temp_2c[unc_map_temp_2c>0])
    tau_array_2c.append(tau_i_2c)
    tau_i_3c = np.nanmean(unc_map_temp_3c[unc_map_temp_3c>0])
    tau_array_3c.append(tau_i_3c)
tau_array_2c = np.array(tau_array_2c)
tau_array_3c = np.array(tau_array_3c)

SI_dice_2c = wjb_functions.dice(SI_mask_2c,GT_mask_2c)
mask_th_dice_2c_array = []
mask_th_dice_3c_array = []
th_range = np.arange(0,11,1)/10
for th in th_range:
    obj_2c = np.where(tau_array_2c<th)[0]
    if obj_2c.size == 0: 
        mask_th_2c = np.zeros([label_image.shape[0],label_image.shape[1]])
    else:
        mask_th_2c_temp = mask_matrix[:,:,obj_2c]
        mask_th_2c = np.sum(mask_th_2c_temp,axis=-1)
        mask_th_2c = wjb_functions.dilation(mask_th_2c, radius)
    obj_3c = np.where(tau_array_3c<th)[0]
    if obj_3c.size == 0: 
        mask_th_3c = np.zeros([label_image.shape[0],label_image.shape[1]])
    else:
        mask_th_3c_temp = mask_matrix[:,:,obj_3c]
        mask_th_3c = np.sum(mask_th_3c_temp,axis=-1)
        mask_th_3c = wjb_functions.dilation(mask_th_3c, radius)
    
    mask_th_dice_2c = wjb_functions.dice(GT_mask_2c, mask_th_2c)
    mask_th_dice_3c = wjb_functions.dice(GT_mask_2c, mask_th_3c)
    mask_th_dice_2c_array.append(mask_th_dice_2c)
    mask_th_dice_3c_array.append(mask_th_dice_3c)
    
    if mask_th_dice_2c > SI_dice_2c: print("RCODDUE")
    if mask_th_dice_3c > SI_dice_2c: print("RCOTTRE")
    
plt.figure()
plt.plot(th_range,np.array(mask_th_dice_2c_array),'b',label='uncertainty map from 2c')
plt.plot(th_range,np.array(mask_th_dice_3c_array),'g',label='uncertainty map from 2c')
plt.axhline(SI_dice_2c,color='r',label="Baseline Dice")
plt.xlim(0,1)
plt.legend()
plt.show()