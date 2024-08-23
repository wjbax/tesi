# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:21:45 2024

@author: willy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:54:30 2024

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
# subset = "val"
# image = "1004289_35.png"
N = 20
radius = 5
c = 3

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_AGOSTO/V05_corretta/"
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
        general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"
        GT_path_2c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
        SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
        GT_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
        SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
        
        
        GT_mask_2c = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        if not GT_mask_2c.any(): continue
        total_images += 1
        SI_mask_2c = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        MC_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "mask/" + image + "/"
        MC_softmax_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "softmax/" + image + "/"
        
        PERT_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "mask/" + image + "/"
        PERT_softmax_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "softmax/" + image + "/"
        
        MC_mask_union_3c_int = wjb_functions.mask_union_gen_3c(MC_path_3c).astype(bool)
        MC_softmax_matrix_3c = wjb_functions.softmax_matrix_gen(MC_softmax_path_3c, np.shape(GT_mask_2c)[:2], 3, N)
        # MC_unc_map_3c_C1 = wjb_functions.binary_entropy_map(MC_softmax_matrix_3c[:,:,1,:])
        # MC_unc_map_3c_C2 = wjb_functions.binary_entropy_map(MC_softmax_matrix_3c[:,:,2,:])
        # MC_unc_map_3c = 0.5*(MC_unc_map_3c_C1+MC_unc_map_3c_C2)
        MC_unc_map_3c = wjb_functions.binary_entropy_map(MC_softmax_matrix_3c[:,:,1,:]+MC_softmax_matrix_3c[:,:,2,:])

        
        PERT_mask_union_3c_int = wjb_functions.mask_union_gen_3c(PERT_path_3c).astype(bool)
        PERT_softmax_matrix_3c = wjb_functions.softmax_matrix_gen(PERT_softmax_path_3c, np.shape(GT_mask_2c)[:2], 3, N)
        # PERT_unc_map_3c_C1 = wjb_functions.binary_entropy_map(PERT_softmax_matrix_3c[:,:,1,:])
        # PERT_unc_map_3c_C2 = wjb_functions.binary_entropy_map(PERT_softmax_matrix_3c[:,:,2,:])
        # PERT_unc_map_3c = 0.5*(PERT_unc_map_3c_C1+PERT_unc_map_3c_C2)
        PERT_unc_map_3c = wjb_functions.binary_entropy_map(PERT_softmax_matrix_3c[:,:,1,:]+PERT_softmax_matrix_3c[:,:,2,:])

        MC_label_image = measure.label(MC_mask_union_3c_int)
        MC_n_objects = MC_label_image.max()
        MC_mask_matrix = np.zeros([MC_label_image.shape[0],MC_label_image.shape[1],MC_n_objects])
        MC_tau_array_3c = []
        for i in range(MC_n_objects):
            current_mask = np.copy(MC_label_image)
            current_mask[current_mask!=i+1] = 0
            max_mask = np.max(current_mask)
            if max_mask == 0: max_mask = 1
            current_mask = current_mask/max_mask
            MC_mask_matrix[:,:,i] = current_mask
            unc_map_temp_3c = np.multiply(current_mask,MC_unc_map_3c)
            tau_i_3c = np.nanmean(unc_map_temp_3c[unc_map_temp_3c>0])
            MC_tau_array_3c.append(tau_i_3c)
        MC_tau_array_3c = np.array(MC_tau_array_3c)
        
        PERT_label_image = measure.label(PERT_mask_union_3c_int)
        PERT_n_objects = PERT_label_image.max()
        PERT_mask_matrix = np.zeros([PERT_label_image.shape[0],PERT_label_image.shape[1],PERT_n_objects])
        PERT_tau_array_3c = []
        for i in range(PERT_n_objects):
            current_mask = np.copy(PERT_label_image)
            current_mask[current_mask!=i+1] = 0
            max_mask = np.max(current_mask)
            if max_mask == 0: max_mask = 1
            current_mask = current_mask/max_mask
            PERT_mask_matrix[:,:,i] = current_mask
            unc_map_temp_3c = np.multiply(current_mask,PERT_unc_map_3c)
            tau_i_3c = np.nanmean(unc_map_temp_3c[unc_map_temp_3c>0])
            PERT_tau_array_3c.append(tau_i_3c)
        PERT_tau_array_3c = np.array(PERT_tau_array_3c)
        
        SI_dice_2c = wjb_functions.dice(SI_mask_2c,GT_mask_2c)
        MC_mask_th_dice_2c_array = []
        MC_mask_th_dice_3c_array = []
        PERT_mask_th_dice_2c_array = []
        PERT_mask_th_dice_3c_array = []
        th_range = np.arange(0,11,1)/10
        
        y=[]
        for th in th_range:
            tau_array_3c = np.copy(MC_tau_array_3c)
            obj_3c = np.where(tau_array_3c<th)[0]
            if obj_3c.size == 0: 
                mask_th_3c = np.zeros([MC_label_image.shape[0],MC_label_image.shape[1]])
                mask_th_3c_no_dilation = np.zeros([MC_label_image.shape[0],MC_label_image.shape[1]])
            else:
                mask_th_3c_temp = MC_mask_matrix[:,:,obj_3c]
                mask_th_3c_no_dilation = np.sum(mask_th_3c_temp,axis=-1)
                mask_th_3c = copy.deepcopy(mask_th_3c_no_dilation)
            
            MC_mask_th_dice_3c = wjb_functions.dice(GT_mask_2c, mask_th_3c)
            MC_mask_th_dice_3c_array.append(MC_mask_th_dice_3c)

            del mask_th_3c
            del tau_array_3c
            
            tau_array_3c = np.copy(PERT_tau_array_3c)
            obj_3c = np.where(tau_array_3c<th)[0]
            if obj_3c.size == 0: 
                mask_th_3c = np.zeros([PERT_label_image.shape[0],PERT_label_image.shape[1]])
                mask_th_3c_no_dilation = np.zeros([PERT_label_image.shape[0],PERT_label_image.shape[1]])
            else:
                mask_th_3c_temp = PERT_mask_matrix[:,:,obj_3c]
                mask_th_3c_no_dilation = np.sum(mask_th_3c_temp,axis=-1)
                mask_th_3c = copy.deepcopy(mask_th_3c_no_dilation)
            
            PERT_mask_th_dice_3c = wjb_functions.dice(GT_mask_2c, mask_th_3c)
            PERT_mask_th_dice_3c_array.append(PERT_mask_th_dice_3c)
            
            dice_array_image = [MC_mask_th_dice_3c,
                                PERT_mask_th_dice_3c]
            
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
    plt.title("Difference between SI dice and max dice per threshold, " + dataset[:-1] + ", subset: " + subset)
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