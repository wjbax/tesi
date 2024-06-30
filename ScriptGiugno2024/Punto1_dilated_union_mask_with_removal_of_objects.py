# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:41:59 2024

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

# diff_type = "PERT"
# diff_type_name = "_perturbation/"
dataset = "Liver HE steatosis/"
# subset = "val"
# image = "1004289_35.png"
N = 20
radius = 5

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V10/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset

dict_unc_map_2c = {
    "image_name": [],
    "SI_dice": [],
    "threshold": [],
    "dice_threshold": []
    }

dict_unc_map_3c = {
    "image_name": [],
    "SI_dice": [],
    "threshold": [],
    "dice_threshold": []
    }

for diff_type in tqdm(["PERT","MC"]):
    if diff_type == "PERT": diff_type_name = "_perturbation"
    if diff_type == "MC": diff_type_name = "_MC"
    for subset in tqdm(["test", "val"]):
        for image in tqdm(os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")):
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
            
            _,GT_mask_3c_int = wjb_functions.mask_splitter(cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE))
            _,SI_mask_3c_int = wjb_functions.mask_splitter(cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE))
            GT_mask_2c = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
            if not GT_mask_2c.any(): continue
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
            SI_dice_3c = wjb_functions.dice(SI_mask_3c_int,GT_mask_3c_int)
            mask_th_dice_2c_array = []
            mask_th_dice_3c_array = []
            mask_th_dice_2c_no_dilation_array = []
            mask_th_dice_3c_no_dilation_array = []
            th_range = np.arange(0,11,1)/10
            for th in th_range:
                obj_2c = np.where(tau_array_2c<th)[0]
                if obj_2c.size == 0: 
                    mask_th_2c = np.zeros([label_image.shape[0],label_image.shape[1]])
                    mask_th_2c_no_dilation = np.zeros([label_image.shape[0],label_image.shape[1]])
                else:
                    mask_th_2c_temp = mask_matrix[:,:,obj_2c]
                    mask_th_2c_no_dilation = np.sum(mask_th_2c_temp,axis=-1)
                    mask_th_2c = wjb_functions.dilation(mask_th_2c_no_dilation, radius)
                obj_3c = np.where(tau_array_3c<th)[0]
                if obj_3c.size == 0: 
                    mask_th_3c = np.zeros([label_image.shape[0],label_image.shape[1]])
                    mask_th_3c_no_dilation = np.zeros([label_image.shape[0],label_image.shape[1]])
                else:
                    mask_th_3c_temp = mask_matrix[:,:,obj_3c]
                    mask_th_3c_no_dilation = np.sum(mask_th_3c_temp,axis=-1)
                    mask_th_3c = wjb_functions.dilation(mask_th_3c_no_dilation, radius)
                
                mask_th_dice_2c = wjb_functions.dice(GT_mask_2c, mask_th_2c)
                mask_th_dice_3c = wjb_functions.dice(GT_mask_2c, mask_th_3c)
                mask_th_dice_2c_array.append(mask_th_dice_2c)
                mask_th_dice_3c_array.append(mask_th_dice_3c)
                mask_th_dice_2c_no_dilation_array.append(wjb_functions.dice(GT_mask_3c_int, mask_th_2c_no_dilation))
                mask_th_dice_3c_no_dilation_array.append(wjb_functions.dice(GT_mask_3c_int, mask_th_3c_no_dilation))
                
                if mask_th_dice_2c > SI_dice_2c: 
                    dict_unc_map_2c["image_name"].append(image)
                    dict_unc_map_2c["SI_dice"].append(SI_dice_2c)
                    dict_unc_map_2c["threshold"].append(th)
                    dict_unc_map_2c["dice_threshold"].append(mask_th_dice_2c)
                if mask_th_dice_3c > SI_dice_2c: 
                    dict_unc_map_3c["image_name"].append(image)
                    dict_unc_map_3c["SI_dice"].append(SI_dice_3c)
                    dict_unc_map_3c["threshold"].append(th)
                    dict_unc_map_3c["dice_threshold"].append(mask_th_dice_3c)
                
            if not os.path.isdir(general_path_to_save + diff_type + "/" + subset + "/"): os.makedirs(general_path_to_save + diff_type + "/" + subset + "/")
            plt.figure(figsize=(18,9))
            plt.suptitle("Immagine " + image[:-4] + ": Dice x Threshold")
            plt.subplot(121)
            plt.title("Case 1: dilated union mask")
            plt.plot(th_range,np.array(mask_th_dice_2c_array),'g',label='uncmap from 2c')
            plt.plot(th_range,np.array(mask_th_dice_3c_array),'b',label='uncmap from 3c')
            plt.axhline(SI_dice_2c,color='r',linestyle='--',label="Baseline Dice")
            plt.xlim(0,1)
            plt.legend()
            plt.subplot(122)
            plt.title("Case 2: NON dilated union mask")
            plt.plot(th_range,np.array(mask_th_dice_2c_no_dilation_array),'g',label='uncmap from 2c (NO DIL)')
            plt.plot(th_range,np.array(mask_th_dice_3c_no_dilation_array), 'b',label='uncmap from 3c (NO DIL)')
            plt.axhline(SI_dice_3c,color='r',linestyle='--',label="Baseline Dice")
            plt.xlim(0,1)
            plt.legend()
            plt.savefig(general_path_to_save + diff_type + "/" + subset + "/" + image)
            plt.close()
            
            df_2c = pd.DataFrame.from_dict(dict_unc_map_2c)
            df_2c.to_csv(general_path_to_save + diff_type + "/" + subset + "/DICE_PER_THRESHOLD_USEFUL_2c.csv",index=False)
            df_3c = pd.DataFrame.from_dict(dict_unc_map_3c)
            df_3c.to_csv(general_path_to_save + diff_type + "/" + subset + "/DICE_PER_THRESHOLD_USEFUL_3c.csv",index=False)