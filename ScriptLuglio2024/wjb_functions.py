# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:03:22 2024

@author: willy
"""

#%% IMPORT

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import measure
import copy
from tqdm import tqdm
import pandas as pd


#%% Funzioni

def mask_union_gen(path_of_3c_directory):
    path = path_of_3c_directory
    temp = cv2.imread(path+"1.png", cv2.IMREAD_GRAYSCALE)
    DIM = np.shape(temp)
    mask_union_int = np.zeros((DIM[0],DIM[1]),dtype=bool)
    for n in os.listdir(path):
        current_mask = cv2.imread(path+n, cv2.IMREAD_GRAYSCALE)
        current_mask = current_mask > 130
        mask_union_int = mask_union_int | current_mask
    return mask_union_int

def mask_union_gen_3c(path_of_3c_directory):
    path = path_of_3c_directory
    temp = cv2.imread(path+"1.png", cv2.IMREAD_GRAYSCALE)
    DIM = np.shape(temp)
    mask_union_int = np.zeros((DIM[0],DIM[1]),dtype=bool)
    for n in os.listdir(path):
        current_mask = cv2.imread(path+n, cv2.IMREAD_GRAYSCALE)
        # current_mask = current_mask > 130
        mask_union_int = mask_union_int | current_mask
    return mask_union_int

def mask_avg_gen(softmax_matrix_3c):
    mean_softmax = np.mean(softmax_matrix_3c, axis=-1)
    mask_avg = np.argmax(mean_softmax, axis=-1)
    values = np.unique(mask_avg)
    if len(values) == 1: return mask_avg
    mask_avg_int = mask_avg > values[-2]
    return mask_avg_int

def mask_avg_gen_3c(softmax_matrix_3c):
    mean_softmax = np.mean(softmax_matrix_3c, axis=-1)
    mask_avg = np.argmax(mean_softmax, axis=-1)
    # mask_avg_int = mask_avg > 1.5
    return mask_avg


def mask_avg_gen_2c(softmax_matrix_3c):
    mean_softmax = np.mean(softmax_matrix_3c, axis=-1)
    mask_avg = np.argmax(mean_softmax, axis=-1)
    return mask_avg

def dice(mask_automatic, mask_manual):
    TP_mask = np.multiply(mask_automatic, mask_manual)
    TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(
        int), TP_mask.astype(int)).astype(bool)
    FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(
        int), TP_mask.astype(int)).astype(bool)
    FN = FN_mask.sum()

    if TP == 0 and FN == 0 and FP == 0:
        dice_ind = np.nan
    else:
        dice_ind = 2*TP/(2*TP+FP+FN)
    return dice_ind

def precision(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()

    if TP==0 and FN==0 and FP==0:
        precision_ind = np.nan
    else:
        precision_ind = TP/(TP+FP)
    return precision_ind

def recall(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()

    if TP==0 and FN==0 and FP==0:
        recall_ind = np.nan
    else:
        recall_ind = TP/(TP+FN)
    return recall_ind


def mask_splitter(mask):
    mask_C1 = np.copy(mask)
    mask_C2 = np.copy(mask)

    mask_C1[mask > 0.8] = 0
    mask_C1[mask_C1 > 0] = 1
    mask_C2[mask < 0.8] = 0
    mask_C2[mask_C2 > 0] = 1
    return mask_C1, mask_C2

def dilation(mask_3c_int,radius): return cv2.dilate(mask_3c_int,np.ones((radius,radius),np.uint8)).astype(bool)

def softmax_matrix_gen(softmax_path, DIM, c, N):
    softmax_matrix = np.zeros((DIM[0], DIM[1], c, N), dtype=np.float32)
    counter = 0
    for num in os.listdir(softmax_path):
        for n_class in range(c):
            st0 = np.float32(
                (np.load(softmax_path + "/" + num)['softmax'])[:, :, n_class])
            softmax_matrix[:, :, n_class, counter] = np.copy(st0)
        counter += 1
    return softmax_matrix

def binary_entropy_map(softmax_matrix):
    p_mean = np.mean(softmax_matrix, axis=2)
    p_mean[p_mean == 0] = 1e-8
    p_mean[p_mean == 1] = 1-1e-8
    HB_pert = -(np.multiply(p_mean, np.log2(p_mean)) +
                np.multiply((1-p_mean), np.log2(1-p_mean)))
    HB_pert[np.isnan(HB_pert)] = np.nanmin(HB_pert)
    return HB_pert

#%% BC
def bhattacharyya_coefficient_2(softmax_matrix_unica):
    radicando_map = np.sqrt(softmax_matrix_unica[:,:,1]*softmax_matrix_unica[:,:,2])
    radicando_map[radicando_map<1e-7] = 1e-7 
    # bc_map = np.log(radicando_map)/np.log(1e-7*1e-7) # è 1e-7^2 perché 1e-7 è sqrt(1e-7^2)
    bc_map = np.log(radicando_map)/np.log(1e-7) # è 1e-7^2 perché 1e-7 è sqrt(1e-7^2)
    return bc_map

def bhattacharyya_coefficient_3(softmax_matrix_unica):
    radicando_map = np.sqrt(softmax_matrix_unica[:,:,0]*softmax_matrix_unica[:,:,1]*softmax_matrix_unica[:,:,2])
    radicando_map[radicando_map<1e-7] = 1e-7 
    # bc_map = np.log(radicando_map)/np.log(1e-7*1e-7)
    bc_map = np.log(radicando_map)/np.log(1e-7)
    return bc_map

def mean_BC_map_2(softmax_matrix_totale):
    radicando_matrix = np.sqrt(softmax_matrix_totale[:,:,1,:]*softmax_matrix_totale[:,:,2,:])
    radicando_matrix[radicando_matrix<1e-7] = 1e-7 
    # bc_map_matrix = np.log(radicando_matrix)/np.log(1e-7*1e-7)
    bc_map_matrix = np.log(radicando_matrix)/np.log(1e-7)
    return np.nanmean(bc_map_matrix,axis=-1)
    
def mean_BC_map_3(softmax_matrix_totale):
    radicando_matrix = np.sqrt(softmax_matrix_totale[:,:,0,:]*softmax_matrix_totale[:,:,1,:]*softmax_matrix_totale[:,:,2,:])
    radicando_matrix[radicando_matrix<1e-7] = 1e-7 
    # bc_map_matrix = np.log(radicando_matrix)/np.log(1e-7*1e-7)
    bc_map_matrix = np.log(radicando_matrix)/np.log(1e-7)
    return np.nanmean(bc_map_matrix,axis=-1)

def mean_softmax_BC_2(softmax_matrix_totale):
    mean_softmax = np.nanmean(softmax_matrix_totale,axis=-1)
    return bhattacharyya_coefficient_2(mean_softmax)

def mean_softmax_BC_3(softmax_matrix_totale):
    mean_softmax = np.nanmean(softmax_matrix_totale,axis=-1)
    return bhattacharyya_coefficient_3(mean_softmax)

def wjmode(unc_map):
    array = np.round(unc_map,2)
    val,counts = np.unique(array, return_counts=True)
    index = np.argmax(counts)
    return val[index]

#%%
def pearsons(array1,array2):
    return pd.DataFrame(np.array([np.array(array1),np.array(array2)]).T).corr()
    