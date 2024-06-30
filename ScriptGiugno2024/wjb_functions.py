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
    mask_avg_int = mask_avg > 1.5
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

def thresholding_dicerecallprecision(mask,
                                     th_range,
                                     unc_map,GT_mask,
                                     SI_mask,max_dice,
                                     max_recall,
                                     max_precision,
                                     path_to_save_plot,
                                     SI_dice,
                                     SI_recall,
                                     SI_precision,
                                     name_of_mask,
                                     image,
                                     th_list_dice,
                                     th_list_recall,
                                     th_list_precision,
                                     max_dice_list,
                                     max_recall_list,
                                     max_precision_list
                                     ):
    dice_per_plot = []
    recall_per_plot = []
    precision_per_plot = []
    
    for th in th_range:
        mask_to_use = mask & (unc_map<th)
        dice_th = dice(GT_mask,mask_to_use)
        dice_per_plot.append(dice_th)
        recall_th = recall(GT_mask,mask_to_use)
        recall_per_plot.append(recall_th)
        precision_th = precision(GT_mask,mask_to_use)
        precision_per_plot.append(precision_th)
        if dice_th > max_dice:
            max_dice = dice_th
            th_max_dice = th
        if recall_th > max_recall:
            max_recall = recall_th
            th_max_recall = th
        if precision_th > max_precision:
            max_precision = precision_th
            th_max_precision = th
            
    dice_per_plot = np.array(dice_per_plot)
    recall_per_plot = np.array(recall_per_plot)
    precision_per_plot = np.array(precision_per_plot)

    # path_to_save_plot = general_savepath + type_of_diff_dict[type_of_diff]["name"] + "/"
    # if not os.path.isdir(path_to_save_plot): os.makedirs(path_to_save_plot)
    
    plt.figure(figsize=(45,5))
    plt.suptitle("Dice, Recall, Precision for image: " + image[:-4] + ", using " + name_of_mask)
    plt.subplot(131)
    plt.plot(th_range,dice_per_plot, 'b', label="dice per th")
    plt.axhline(SI_dice, color='r', label="Single Inference dice")
    # plt.axhline(dice_th, color='g', label="Total Mask dice")
    plt.xlim(0,1)
    plt.xlabel("Threshold")
    plt.title("Dice per threshold")
    plt.legend()
    plt.subplot(132)
    plt.plot(th_range,recall_per_plot, 'b', label="recall per th")
    plt.axhline(SI_recall, color='r', label="Single Inference recall")
    # plt.axhline(recall_th, color='g', label="Total Mask recall")
    plt.xlim(0,1)
    plt.xlabel("Threshold")
    plt.title("Recall per threshold")
    plt.legend()
    plt.subplot(133)
    plt.plot(th_range,precision_per_plot, 'b', label="precision per th")
    plt.axhline(SI_precision, color='r', label="Single Inference precision")
    # plt.axhline(precision_th, color='g', label="Total Mask precision")
    plt.xlim(0,1)
    plt.xlabel("Threshold")
    plt.title("Precision per threshold")
    plt.legend()
    plt.savefig(path_to_save_plot + image[:-4] + "_subplot_per_threshold.png")
    # plt.savefig(path_to_save_plot + image[:-4] + "_" + name_of_mask + "_subplot_per_threshold.png")
    plt.close()
    
    if max_dice == 0: th_max_dice = 0 
    if max_recall == 0: th_max_recall = 0 
    if max_precision == 0: th_max_precision = 0 
     
    th_list_dice.append(th_max_dice)
    th_list_recall.append(th_max_recall)
    th_list_precision.append(th_max_precision)
    
    max_dice_list.append(max_dice)
    max_recall_list.append(max_recall)
    max_precision_list.append(max_precision)
    return th_list_dice, th_list_recall, th_list_precision, max_dice_list, max_recall_list, max_precision_list