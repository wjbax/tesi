# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:16:37 2024

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
image = "1004302_88.png"
N = 20
# radius = 5
c = 3

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V23/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

image_path = os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")
GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
OR_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "image/" + image
OR_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "image/" + image
SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image
SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
diff_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
diff_path_3c = general_results_path_3c + diff_type_name + subset + "/" + "mask/" + image + "/"
softmax_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"

#%%
OR_image = cv2.imread(OR_path_3c)
OR_image = cv2.cvtColor(OR_image, cv2.COLOR_BGR2RGB)
GT_mask = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
GT_mask_C1,GT_mask_C2 = wjb_functions.mask_splitter(GT_mask)
SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
SI_mask_C1,SI_mask_C2 = wjb_functions.mask_splitter(SI_mask)
DIM = np.shape(GT_mask)[:2]

#%%
plt.imshow(OR_image)

#%%
softmax_matrix_PERT = wjb_functions.softmax_matrix_gen(softmax_path_3c, DIM, c, N)
SI_dice = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
# max_dice_PERT = 0
# for i in range(20):
#     current_softmax = softmax_matrix_PERT[:,:,:,i]
#     current_mask = np.argmax(current_softmax,axis=-1)
#     current_mask_C1,current_mask_C2 = wjb_functions.mask_splitter(current_mask)
#     dice_cm = wjb_functions.dice(current_mask_C2,GT_mask_C2)
#     print(dice_cm)
#     if dice_cm > max_dice_PERT:
#         max_dice_PERT = dice_cm
#         best_mask_PERT = copy.deepcopy(current_mask_C2)
        
#%%
max_dice_PERT = 0
for i in range(21):
    if i == 0: continue
    current_mask = cv2.imread(diff_path_3c + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
    current_mask_C1,current_mask_C2 = wjb_functions.mask_splitter(current_mask)
    dice_cm = wjb_functions.dice(current_mask_C2,GT_mask_C2)
    print(dice_cm)
    if dice_cm > max_dice_PERT:
        max_dice_PERT = dice_cm
        best_mask_PERT = copy.deepcopy(current_mask_C2)
    
#%%

diff_path_3c_MC = general_results_path_3c + "_MC" + "/" + subset + "/" + "mask/" + image + "/"
max_dice_MC = 0
for i in range(20):
    if i == 0: continue
    current_mask = cv2.imread(diff_path_3c_MC + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
    current_mask_C1,current_mask_C2 = wjb_functions.mask_splitter(current_mask)
    dice_cm = wjb_functions.dice(current_mask_C2,GT_mask_C2)
    print(dice_cm)
    if dice_cm > max_dice_MC:
        max_dice_MC = dice_cm
        best_mask_MC = copy.deepcopy(current_mask_C2)
    
    
#%%

binent_map = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,2,:])
BC_map = wjb_functions.mean_softmax_BC_3(softmax_matrix_PERT)

#%%
plt.figure()
plt.subplot(241)
plt.imshow(OR_image)
plt.subplot(242)
plt.imshow(SI_mask_C2)
plt.subplot(243)
plt.imshow(best_mask_PERT)
plt.subplot(244)
plt.imshow(best_mask_MC)
plt.subplot(245)
plt.imshow(GT_mask_C2)
plt.subplot(246)
plt.imshow(binent_map)
plt.subplot(247)
plt.imshow(BC_map)
plt.subplot(248)
plt.show()











