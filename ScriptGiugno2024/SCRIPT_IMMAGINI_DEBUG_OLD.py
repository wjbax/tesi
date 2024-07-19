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
from skimage.segmentation import mark_boundaries
import copy
from tqdm import tqdm
import wjb_functions
import pandas as pd

#%%

diff_type = "PERT"
diff_type_name = "_perturbation/"
dataset = "Renal PAS glomeruli/"
subset = "test"
image = "1004763_1.png"
N = 20
# radius = 5
c = 3

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V26/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
# general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
# general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"

# image_path = os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")
# GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
# OR_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "image/" + image
# OR_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "image/" + image
# GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image

image_path = os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/")
GT_path_2c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
OR_path_2c = general_dataset_path + "DATASET/" + subset + "/" + "image/" + image
OR_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "image/" + image
SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
GT_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
diff_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
diff_path_3c = general_results_path_3c + diff_type_name + subset + "/" + "mask/" + image + "/"
softmax_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"

#%%
OR_image = cv2.imread(OR_path_3c)
OR_image = cv2.cvtColor(OR_image, cv2.COLOR_BGR2RGB)
GT_mask = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)
GT_mask_C1,GT_mask_C2 = wjb_functions.mask_splitter(GT_mask/255)
SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)
SI_mask_C1,SI_mask_C2 = wjb_functions.mask_splitter(SI_mask/255)
DIM = np.shape(GT_mask)[:2]

#%%

SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)

GT_masked_image_C1 = mark_boundaries(OR_image, GT_mask_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
# plt.imshow(GT_masked_image_C1)

GT_masked_image_C2 = mark_boundaries(OR_image, GT_mask_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
# plt.imshow(GT_masked_image_C2)

SI_masked_image_C1 = mark_boundaries(OR_image, SI_mask_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
# plt.imshow(SI_masked_image_C1)

SI_masked_image_C2 = mark_boundaries(OR_image, SI_mask_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
# plt.imshow(SI_masked_image_C2)

#%%
softmax_matrix_PERT = wjb_functions.softmax_matrix_gen(softmax_path_3c, DIM, c, N)

#%%
max_dice_PERT_C2 = 0
max_dice_PERT_C1 = 0
dice_PERT_C1 = []
dice_PERT_C2 = []
TOT_dice_C1 = []
TOT_dice_C2 = []
for i in range(21):
    if i == 0: continue
    print(i)
    current_mask = cv2.imread(diff_path_3c + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
    current_mask_C1,current_mask_C2 = wjb_functions.mask_splitter(current_mask)
    dice_cm_C2 = wjb_functions.dice(current_mask_C2,GT_mask_C2)
    if dice_cm_C2 > max_dice_PERT_C2:
        print(dice_cm_C2)
        max_dice_PERT_C2 = dice_cm_C2
        best_mask_PERT_C2 = copy.deepcopy(current_mask_C2)
    dice_cm_C1 = wjb_functions.dice(current_mask_C1,GT_mask_C1)
    if dice_cm_C1 > max_dice_PERT_C1:
        print(dice_cm_C1)
        max_dice_PERT_C1 = dice_cm_C1
        best_mask_PERT_C1 = copy.deepcopy(current_mask_C1)
    dice_PERT_C1.append(dice_cm_C1)
    dice_PERT_C2.append(dice_cm_C2)
    TOT_dice_C1.append(dice_cm_C1)
    TOT_dice_C2.append(dice_cm_C2)

#%%
diff_path_3c_MC = general_results_path_3c + "_MC/" + subset + "/" + "mask/" + image + "/"

max_dice_MC_C1 = 0
max_dice_MC_C2 = 0
dice_MC_C1 = []
dice_MC_C2 = []
for i in range(20):
    if i == 0: continue
    current_mask = cv2.imread(diff_path_3c_MC + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
    current_mask_C1,current_mask_C2 = wjb_functions.mask_splitter(current_mask)
    dice_cm_C2 = wjb_functions.dice(current_mask_C2,GT_mask_C2)
    if dice_cm_C2 > max_dice_MC_C2:
        print(dice_cm_C2)
        max_dice_MC_C2 = dice_cm_C2
        best_mask_MC_C2 = copy.deepcopy(current_mask_C2)
    dice_cm_C1 = wjb_functions.dice(current_mask_C1,GT_mask_C1)
    if dice_cm_C1 > max_dice_MC_C1:
        print(dice_cm_C1)
        max_dice_MC_C1 = dice_cm_C1
        best_mask_MC_C1 = copy.deepcopy(current_mask_C1)
    dice_MC_C1.append(dice_cm_C1)
    dice_MC_C2.append(dice_cm_C2)
    TOT_dice_C1.append(dice_cm_C1)
    TOT_dice_C2.append(dice_cm_C2)
#%%

PERT_masked_image_C1 = mark_boundaries(OR_image, best_mask_PERT_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
PERT_masked_image_C2 = mark_boundaries(OR_image, best_mask_PERT_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)

MC_masked_image_C1 = mark_boundaries(OR_image, best_mask_MC_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
MC_masked_image_C2 = mark_boundaries(OR_image, best_mask_MC_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)

#%%
binent_map_C1 = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
binent_map_C2 = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,2,:])
BC_map = wjb_functions.mean_softmax_BC_3(softmax_matrix_PERT)

#%%

data_C1 = [SI_dice_C1, dice_MC_C1, dice_PERT_C1, TOT_dice_C1]
data_C2 = [SI_dice_C2, dice_MC_C2, dice_PERT_C2, TOT_dice_C2]

x_label_C1 = ['Single Inference Dice', 'Monte Carlo Dices', 'Perturbations Dices', 'Combined Dices']
x_label_C2 = ['Single Inference Dice', 'Monte Carlo Dices', 'Perturbations Dices', 'Combined Dices']

#%%

mask_matrix = wjb_functions.mask_matrix_gen_3c(diff_path_2c,DIM,N)/255            
mask_matrix_C1,mask_matrix_C2 = wjb_functions.mask_splitter(mask_matrix)

dice_mat_PERT_C1 = wjb_functions.intradice_mat(mask_matrix_C1)
dice_mat_PERT_C2 = wjb_functions.intradice_mat(mask_matrix_C2)

del mask_matrix
del mask_matrix_C1
del mask_matrix_C2
mask_matrix = wjb_functions.mask_matrix_gen_3c(diff_path_3c_MC,DIM,N)/255            
mask_matrix_C1,mask_matrix_C2 = wjb_functions.mask_splitter(mask_matrix)

dice_mat_MC_C1 = wjb_functions.intradice_mat(mask_matrix_C1)
dice_mat_MC_C2 = wjb_functions.intradice_mat(mask_matrix_C2)

#%%
plt.figure()
plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4], ", class 1")
plt.subplot(241)
plt.title("Ground Truth")
plt.imshow(GT_masked_image_C1)
plt.subplot(242)
plt.title("Single Inference")
plt.imshow(SI_masked_image_C1)
plt.subplot(243)
plt.title("Best Perturbation Mask")
plt.imshow(PERT_masked_image_C1)
plt.subplot(244)
plt.title("Best Montecarlo Mask")
plt.imshow(MC_masked_image_C1)
plt.subplot(245)
plt.title("Binary Entropy Uncertainty Map")
plt.imshow(binent_map_C1)
plt.colormap()
plt.subplot(246)
plt.title("Bhattacharyya Uncertainty Map")
plt.imshow(BC_map)
plt.colormap()
plt.subplot(247)
plt.title("Boxplot dices pert and MC")
plt.boxplot(data_C1)
plt.ylabel('DICE')
plt.xticks([1,2,3,4], x_label_C1)
plt.subplot(248)
plt.subplot(121)
plt.title("Intradice matrix MC")
plt.imshow(dice_mat_MC_C1)
plt.subplot(122)
plt.title("Intradice matrix PERT")
plt.imshow(dice_mat_PERT_C1)
plt.show()
# plt.savefig()
# plt.close()