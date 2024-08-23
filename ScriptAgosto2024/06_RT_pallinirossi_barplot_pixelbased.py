# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:32:30 2024

@author: willy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:25:29 2024

@author: willy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:03:15 2024

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

# diff_type = "PERT"
# diff_type_name = "_perturbation/"
dataset = "Renal PAS tubuli/"
c = 3
# subset = "val"
# image = "1004289_35.png"
N = 20
radius = 5

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_AGOSTO/V06_corretta/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset

better_dices = 0
total_images = 0
for subset in tqdm(["test", "val"]):
    
    dict_per_plots = {
        "image_number": [],
        "image_name": [],
        "SI_dice": [],
        "diff_dice": [],
        "max_dice": [],
        "threshold_max_dice": [],
        "flag": []
        }
    
    for image in tqdm(os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/")):
        
        general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
        GT_path_2c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
        SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
        
        MC_path_2c = general_results_path_2c + "_MC/" + subset + "/" + "mask/" + image + "/"
        MC_softmax_path_2c = general_results_path_2c + "_MC/" + subset + "/" + "softmax/" + image + "/"
        
        PERT_path_2c = general_results_path_2c + "_perturbation/" + subset + "/" + "mask/" + image + "/"
        PERT_softmax_path_2c = general_results_path_2c + "_perturbation/" + subset + "/" + "softmax/" + image + "/"
        
        GT_mask_2c = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        if not GT_mask_2c.any(): continue
        DIM = np.shape(GT_mask_2c)[:2]
        total_images += 1
        SI_mask_2c = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        PERT_mask_2c = wjb_functions.mask_union_gen_3c(PERT_path_2c).astype(bool)
        MC_mask_2c = wjb_functions.mask_union_gen_3c(MC_path_2c).astype(bool)
        
        MC_softmax_matrix = wjb_functions.softmax_matrix_gen(MC_softmax_path_2c, DIM, c, N)
        PERT_softmax_matrix = wjb_functions.softmax_matrix_gen(PERT_softmax_path_2c, DIM, c, N)
        
        # MC_unc_map_C1 = wjb_functions.binary_entropy_map(MC_softmax_matrix[:,:,1,:])
        # PERT_unc_map_C1 = wjb_functions.binary_entropy_map(PERT_softmax_matrix[:,:,1,:])
        # MC_unc_map_C2 = wjb_functions.binary_entropy_map(MC_softmax_matrix[:,:,2,:])
        # PERT_unc_map_C2 = wjb_functions.binary_entropy_map(PERT_softmax_matrix[:,:,2,:])
        
        MC_unc_map = wjb_functions.binary_entropy_map(MC_softmax_matrix[:,:,1,:]+MC_softmax_matrix[:,:,2,:])
        PERT_unc_map = wjb_functions.binary_entropy_map(PERT_softmax_matrix[:,:,1,:]+PERT_softmax_matrix[:,:,2,:])

        SI_dice_2c = wjb_functions.dice(SI_mask_2c,GT_mask_2c)
        MC_mask_th_dice_2c_array = []
        PERT_mask_th_dice_2c_array = []
        th_range = np.arange(0,11,1)/10
        
        y=[]
        for th in th_range:
            PERT_current_mask = PERT_mask_2c & (PERT_unc_map<=th)
            MC_current_mask = MC_mask_2c & (MC_unc_map<=th)
            
            PERT_mask_th_dice_2c = wjb_functions.dice(GT_mask_2c, PERT_current_mask)
            PERT_mask_th_dice_2c_array.append(PERT_mask_th_dice_2c)
            
            MC_mask_th_dice_2c = wjb_functions.dice(GT_mask_2c, MC_current_mask)
            MC_mask_th_dice_2c_array.append(MC_mask_th_dice_2c)
            
            dice_array_image = [MC_mask_th_dice_2c,
                                PERT_mask_th_dice_2c]
            
            max_dice = np.max(dice_array_image)
            y.append(max_dice)
            
        if not os.path.isdir(general_path_to_save + subset + "/"): os.makedirs(general_path_to_save + subset + "/")
        plt.figure(figsize=(10,8))
        plt.title('Max dice image ' + image + ',  from subset: ' + subset + " from dataset: " + dataset[:-1])
        plt.plot(th_range,y,'b', label='Max dice from PERT or MC')
        plt.axhline(SI_dice_2c,color='r',linestyle='--',label="Baseline Dice")
        plt.xlabel("Threshold on uncertainty map")
        plt.ylabel("Max dice (between MC and PERT) per threshold")
        plt.xlim(0,1)
        plt.ylim(0)
        plt.legend()
        plt.savefig(general_path_to_save + subset + "/" + image)
        plt.close()
        
        if np.max(y) > SI_dice_2c: 
            better_dices += 1
            print("Image " + image + ',  from subset: ' + subset + " has a better Dice!")
            
        dict_per_plots["image_number"].append(total_images)
        dict_per_plots["image_name"].append(image)
        dict_per_plots["SI_dice"].append(SI_dice_2c)
        dict_per_plots["diff_dice"].append(SI_dice_2c-np.max(y))
        dict_per_plots["max_dice"].append(np.max(y))
        dict_per_plots["threshold_max_dice"].append(th_range[np.where(y==np.max(y))][0])
        if np.max(y) > SI_dice_2c:
            dict_per_plots["flag"].append(True)
        else:
            dict_per_plots["flag"].append(False)
        
    x = np.arange(0,len(dict_per_plots['threshold_max_dice']),1)
    plt.figure(figsize=(10,8))
    plt.title("Optimal Threshold for dice")
    Y = np.squeeze(np.array(dict_per_plots["threshold_max_dice"]))
    plt.scatter(x,Y,color='b',label="Optimal Threhsold - dice")
    plt.scatter(x[np.where(np.array(dict_per_plots['flag'])==True)],
                np.squeeze(np.array(dict_per_plots['threshold_max_dice'])[np.where(np.array(dict_per_plots['flag'])==True)]),color='r',
                label='At this threshold, dice(th) is GREATER than baseline dice')
    plt.xlabel("Image number")
    plt.ylabel("Opt th - dice")
    plt.legend()
    plt.savefig(general_path_to_save + subset + "/pallinirossi.png")
    plt.close()
    del Y
    
    plt.figure(figsize=(10,8))
    plt.title("Difference between SI dice and max dice per threshold, " + dataset[:-1] + ", subset: " + subset + " (pixel based)")
    Y = np.squeeze(np.array(dict_per_plots["diff_dice"]))
    colors = ['red' if value > 0 else 'green' for value in Y]
    plt.bar(x,Y, color=colors)
    plt.axhline(0)
    plt.xlabel("Image number")
    plt.ylabel("Differences between dices")
    plt.savefig(general_path_to_save + subset + "/barplot_dices.png")
    plt.close()
            
perc_miglioramento = better_dices / total_images
perc_miglioramento = np.round(perc_miglioramento,3)*100
print("E' stato migliorato il " + str(perc_miglioramento)[:4] + "% delle immagini")