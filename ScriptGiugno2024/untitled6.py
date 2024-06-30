# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:36:12 2024

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

Perturbations = {
    "results" : "RESULTS_perturbation",
    "name" : "PERT"
    }

Montecarlo = {
    "results": "RESULTS_MC",
    "name": "MC"
    }

type_of_diff_dict = {
    "Perturbations": Perturbations,
    "Montecarlo": Montecarlo
    }

#%%
N = 20
general_savepath = "D:/DATASET_Tesi_marzo2024_RESULTS_V9/"
image_path = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_2classes/RESULTS/test/mask/"

for name_of_mask in ["union_mask", "avg_mask", "SI_mask"]:
    
    for type_of_diff in tqdm(type_of_diff_dict):
        
        th_list_dice = []
        th_list_recall = []
        th_list_precision = []
        max_dice_list = []
        max_recall_list = []
        max_precision_list = []
        image_list = []
        
        for image in tqdm(os.listdir(image_path)):
            
            SI_path = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_2classes/RESULTS/test/mask/" + image
            GT_path = 'D:/DATASET_Tesi_marzo2024/Liver HE steatosis/DATASET_2classes/test/manual/' + image
            path_of_images = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_2classes/" + type_of_diff_dict[type_of_diff]["results"] + "/test/mask/" + image + "/"
            softmax_matrix_path = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_2classes/" + type_of_diff_dict[type_of_diff]["results"] + "/test/softmax/" + image + "/"
            
            SI_mask = cv2.imread(SI_path, cv2.IMREAD_GRAYSCALE).astype(bool)
            GT_mask = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE).astype(bool)
            if not GT_mask.any(): continue
            image_list.append(image)
            
            if name_of_mask == "union_mask": mask = wjb_functions.mask_union_gen(path_of_images)
            if name_of_mask == "avg_mask": 
                sm_path_3c = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_3classes/" + type_of_diff_dict[type_of_diff]["results"] + "/test/softmax/" + image + "/"
                softmax_matrix_3c = wjb_functions.softmax_matrix_gen(sm_path_3c, np.shape(SI_mask)[:2], 3, N)
                mask = wjb_functions.mask_avg_gen(softmax_matrix_3c)
            if name_of_mask == "SI_mask": mask = SI_mask
            
            softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_matrix_path, np.shape(SI_mask)[:2], 2, N)
            
            max_dice = 0; max_precision = 0; max_recall = 0;
            
            th_range = np.arange(0,21,1)/20
            
            SI_dice = wjb_functions.dice(SI_mask,GT_mask)
            SI_recall = wjb_functions.recall(SI_mask,GT_mask)
            SI_precision = wjb_functions.precision(SI_mask,GT_mask)
            
            unc_map = np.nanmean(wjb_functions.binary_entropy_map(softmax_matrix[:,:,1,:]),axis=-1)
            
            
            path_to_save_plot = general_savepath + type_of_diff_dict[type_of_diff]["name"] + "/" + name_of_mask + "/"
            if not os.path.isdir(path_to_save_plot): os.makedirs(path_to_save_plot)
            
            [th_list_dice, 
            th_list_recall, 
            th_list_precision, 
            max_dice_list, 
            max_recall_list, 
            max_precision_list] = wjb_functions.thresholding_dicerecallprecision(
                mask,
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
                )
            
            # dice_per_plot = []
            # recall_per_plot = []
            # precision_per_plot = []
            
            # for th in th_range:
            #     mask_to_use = union_mask | (unc_map>th)
            #     dice_th = wjb_functions.dice(GT_mask,mask_to_use)
            #     dice_per_plot.append(dice_th)
            #     recall_th = wjb_functions.recall(GT_mask,mask_to_use)
            #     recall_per_plot.append(recall_th)
            #     precision_th = wjb_functions.precision(GT_mask,mask_to_use)
            #     precision_per_plot.append(precision_th)
            #     if dice_th > max_dice:
            #         max_dice = dice_th
            #         th_max_dice = th
            #     if recall_th > max_recall:
            #         max_recall = recall_th
            #         th_max_recall = th
            #     if precision_th > max_precision:
            #         max_precision = precision_th
            #         th_max_precision = th
                    
            # dice_per_plot = np.array(dice_per_plot)
            # recall_per_plot = np.array(recall_per_plot)
            # precision_per_plot = np.array(precision_per_plot)
        
            # path_to_save_plot = general_savepath + type_of_diff_dict[type_of_diff]["name"] + "/"
            # if not os.path.isdir(path_to_save_plot): os.makedirs(path_to_save_plot)
            
            # plt.figure(figsize=(45,7))
            # plt.subplot(131)
            # plt.plot(th_range,dice_per_plot, 'b', label="dice per th")
            # plt.axhline(SI_dice, color='r', label="Single Inference dice")
            # plt.xlabel("Threshold")
            # plt.legend()
            # plt.subplot(132)
            # plt.plot(th_range,recall_per_plot, 'b', label="recall per th")
            # plt.axhline(SI_recall, color='r', label="Single Inference recall")
            # plt.xlabel("Threshold")
            # plt.legend()
            # plt.subplot(133)
            # plt.plot(th_range,precision_per_plot, 'b', label="precision per th")
            # plt.axhline(SI_precision, color='r', label="Single Inference precision")
            # plt.xlabel("Threshold")
            # plt.legend()
            # plt.savefig(path_to_save_plot + image[:-4] + "_subplot_per_threshold.png")
            # plt.close()
             
            # th_list_dice.append(th_max_dice)
            # th_list_recall.append(th_max_recall)
            # th_list_precision.append(th_max_precision)
            
            # max_dice_list.append(max_dice)
            # max_recall_list.append(max_recall)
            # max_precision_list.append(max_precision)
            
        dict_for_dataframe = {
            
            "Image_list": image_list,
            "Th_max_dice": th_list_dice,
            "Max_dices": max_dice_list,
            "Th_max_recall": th_list_recall,
            "Max_recalls": max_recall_list,
            "Th_max_precision": th_list_precision,
            "Max_precisions": max_precision_list
            }
            
        savepath_df = general_savepath + type_of_diff_dict[type_of_diff]["name"] + "/" + name_of_mask
        if not os.path.isdir(savepath_df): os.makedirs(savepath_df)
        df = pd.DataFrame.from_dict(dict_for_dataframe)
        df.to_csv(savepath_df + "/dati_dicerecallprecision.csv",index=False)