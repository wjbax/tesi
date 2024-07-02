# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:36:41 2024

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

def bhattacharyya_coefficient_2(softmax_matrix_unica):
    radicando_map = np.sqrt(softmax_matrix_unica[:,:,1]*softmax_matrix_unica[:,:,2])
    radicando_map[radicando_map<1e-7] = 1e-7 
    bc_map = np.log(radicando_map)/np.log(np.sqrt(1e-7*1e-7))
    return bc_map

def bhattacharyya_coefficient_3(softmax_matrix_unica):
    radicando_map = np.sqrt(softmax_matrix_unica[:,:,0]*softmax_matrix_unica[:,:,1]*softmax_matrix_unica[:,:,2])
    radicando_map[radicando_map<1e-7] = 1e-7 
    bc_map = np.log(radicando_map)/np.log(np.sqrt(1e-7*1e-7))
    return bc_map

def mean_BC_map_2(softmax_matrix_totale):
    radicando_matrix = np.sqrt(softmax_matrix_totale[:,:,1,:]*softmax_matrix_totale[:,:,2,:])
    radicando_matrix[radicando_matrix<1e-7] = 1e-7 
    bc_map_matrix = np.log(radicando_matrix)/np.log(np.sqrt(1e-7*1e-7))
    return np.nanmean(bc_map_matrix,axis=-1)
    
def mean_BC_map_3(softmax_matrix_totale):
    radicando_matrix = np.sqrt(softmax_matrix_totale[:,:,0,:]*softmax_matrix_totale[:,:,1,:]*softmax_matrix_totale[:,:,2,:])
    radicando_matrix[radicando_matrix<1e-7] = 1e-7 
    bc_map_matrix = np.log(radicando_matrix)/np.log(np.sqrt(1e-7*1e-7))
    return np.nanmean(bc_map_matrix,axis=-1)

def mean_softmax_BC_2(softmax_matrix_totale):
    mean_softmax = np.nanmean(softmax_matrix_totale,axis=-1)
    return bhattacharyya_coefficient_2(mean_softmax)

def mean_softmax_BC_3(softmax_matrix_totale):
    mean_softmax = np.nanmean(softmax_matrix_totale,axis=-1)
    return bhattacharyya_coefficient_3(mean_softmax)

diff_type = "PERT"
diff_type_name = "_perturbation/"
dataset = "Renal PAS glomeruli/"
subset = "test"
image = "1004761_25.png"
N = 20
# radius = 5
c = 3

#%%
general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V11/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"

GT_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
diff_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"

dataset_name = dataset[:-1] + "_3c/"
GT_mask = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
GT_mask_C1, GT_mask_C2 = wjb_functions.mask_splitter(GT_mask)

SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
SI_mask_C1, SI_mask_C2 = wjb_functions.mask_splitter(SI_mask)

DIM = np.shape(GT_mask)[:2]
softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_3c, DIM, c, N)

bc_map = bhattacharyya_coefficient_2(softmax_matrix[:,:,:,0])
plt.imshow(bc_map)

bc_map = bhattacharyya_coefficient_3(softmax_matrix[:,:,:,0])
plt.imshow(bc_map)

#%%

bc_map_2_mean = mean_BC_map_2(softmax_matrix)
bc_map_3_mean = mean_BC_map_3(softmax_matrix)

#%%

mean_bc_map_2 = mean_softmax_BC_2(softmax_matrix)
mean_bc_map_3 = mean_softmax_BC_3(softmax_matrix)

#%% MEAN SOFTMAX -> BC (3 CLASSES)
dice_C1_mbc3 = []
dice_C2_mbc3 = []
th_range = np.arange(0,11,1)/10
unc_map = mean_bc_map_3
SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
print("dice_C1 = {d1}, dice_C2 = {d2}".format(d1=SI_dice_C1,d2=SI_dice_C2))
for th in th_range:
    SI_red = SI_mask * (unc_map >= th)
    SI_red_C1, SI_red_C2 = wjb_functions.mask_splitter(SI_red)
    dice_C1 = wjb_functions.dice(SI_red_C1,GT_mask_C1)
    dice_C2 = wjb_functions.dice(SI_red_C2,GT_mask_C2)
    dice_C1_mbc3.append(dice_C1)
    dice_C2_mbc3.append(dice_C2)
    print("Th = {th}: dice_C1 = {d1}, dice_C2 = {d2}".format(th=th,d1=dice_C1,d2=dice_C2))
    if dice_C1 > SI_dice_C1: print("DICE C1 IS GREATER")
    if dice_C2 > SI_dice_C2: print("DICE C2 IS GREATER")
    
#%% MEAN SOFTMAX -> BC (2 CLASSES)
dice_C1_mbc2 = []
dice_C2_mbc2 = []
th_range = np.arange(0,11,1)/10
unc_map = mean_bc_map_2
SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
print("dice_C1 = {d1}, dice_C2 = {d2}".format(d1=SI_dice_C1,d2=SI_dice_C2))
for th in th_range:
    SI_red = SI_mask * (unc_map >= th)
    SI_red_C1, SI_red_C2 = wjb_functions.mask_splitter(SI_red)
    dice_C1 = wjb_functions.dice(SI_red_C1,GT_mask_C1)
    dice_C2 = wjb_functions.dice(SI_red_C2,GT_mask_C2)
    dice_C1_mbc2.append(dice_C1)
    dice_C2_mbc2.append(dice_C2)
    print("Th = {th}: dice_C1 = {d1}, dice_C2 = {d2}".format(th=th,d1=dice_C1,d2=dice_C2))
    if dice_C1 > SI_dice_C1: print("DICE C1 IS GREATER")
    if dice_C2 > SI_dice_C2: print("DICE C2 IS GREATER")
    
#%% BC ON TOTAL SOFTMAX -> THEN MEAN OF BCs on 2 classes
dice_C1_bc2m = []
dice_C2_bc2m = []
th_range = np.arange(0,11,1)/10
unc_map = bc_map_2_mean
SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
print("dice_C1 = {d1}, dice_C2 = {d2}".format(d1=SI_dice_C1,d2=SI_dice_C2))
for th in th_range:
    SI_red = SI_mask * (unc_map >= th)
    SI_red_C1, SI_red_C2 = wjb_functions.mask_splitter(SI_red)
    dice_C1 = wjb_functions.dice(SI_red_C1,GT_mask_C1)
    dice_C2 = wjb_functions.dice(SI_red_C2,GT_mask_C2)
    dice_C1_bc2m.append(dice_C1)
    dice_C2_bc2m.append(dice_C2)
    print("Th = {th}: dice_C1 = {d1}, dice_C2 = {d2}".format(th=th,d1=dice_C1,d2=dice_C2))
    if dice_C1 > SI_dice_C1: print("DICE C1 IS GREATER")
    if dice_C2 > SI_dice_C2: print("DICE C2 IS GREATER")

#%% BC ON TOTAL SOFTMAX -> THEN MEAN OF BCs on 3 classes
dice_C1_bc3m = []
dice_C2_bc3m = []
th_range = np.arange(0,11,1)/10
unc_map = bc_map_3_mean
SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
print("dice_C1 = {d1}, dice_C2 = {d2}".format(d1=SI_dice_C1,d2=SI_dice_C2))
for th in th_range:
    SI_red = SI_mask * (unc_map >= th)
    SI_red_C1, SI_red_C2 = wjb_functions.mask_splitter(SI_red)
    dice_C1 = wjb_functions.dice(SI_red_C1,GT_mask_C1)
    dice_C2 = wjb_functions.dice(SI_red_C2,GT_mask_C2)
    dice_C1_bc3m.append(dice_C1)
    dice_C2_bc3m.append(dice_C2)
    print("Th = {th}: dice_C1 = {d1}, dice_C2 = {d2}".format(th=th,d1=dice_C1,d2=dice_C2))
    if dice_C1 > SI_dice_C1: print("DICE C1 IS GREATER")
    if dice_C2 > SI_dice_C2: print("DICE C2 IS GREATER")
    
#%%
dice_C1_mbc3 = np.array(dice_C1_mbc3)
dice_C2_mbc3 = np.array(dice_C2_mbc3)
dice_C1_mbc2 = np.array(dice_C1_mbc2)
dice_C2_mbc2 = np.array(dice_C2_mbc2)

dice_C1_bc3m = np.array(dice_C1_bc3m)
dice_C2_bc3m = np.array(dice_C2_bc3m)
dice_C1_bc2m = np.array(dice_C1_bc2m)
dice_C2_bc2m = np.array(dice_C2_bc2m)

#%%
C1_data = np.array([dice_C1_mbc3,dice_C1_bc3m,dice_C1_mbc2,dice_C1_bc2m]).T
C2_data = np.array([dice_C2_mbc3,dice_C2_bc3m,dice_C2_mbc2,dice_C2_bc2m]).T

#%%

plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(bc_map_2_mean)
plt.title("BC on total softmax, then mean, 2 classes")
plt.subplot(222)
plt.imshow(bc_map_3_mean)
plt.title("BC on total softmax, then mean, 3 classes")
plt.subplot(223)
plt.imshow(mean_bc_map_2)
plt.title("mean of softmax, then BC, 2 classes")
plt.subplot(224)
plt.imshow(mean_bc_map_3)
plt.title("mean of softmax, then BC, 3 classes")

#%% USO 3 CLASSI DIRETTAMENTE