# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:17:38 2024

@author: willy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import cv2
import os
from skimage import measure
import copy
from tqdm import tqdm
import wjb_functions
import pandas as pd

# diff_type = "PERT"
# diff_type_name = "_perturbation/"
dataset = "Liver HE steatosis/"
subset = "test"
image = "1004289_35.png"
N = 20
radius = 5

colors_for_plot = [                               
    to_rgb('#006400'),
    to_rgb('#bc8f8f'),
    to_rgb('#ffd700'),
    to_rgb('#0000cd'),
    to_rgb('#00ff00'),
    to_rgb('#00ffff'),
    to_rgb('#1e90ff'),
    to_rgb('#ff1493')]

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V28/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset

general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"


for subset in tqdm(["train"]):
    
    # if subset == "test": continue
    
    PERT_dice_array_SI_and_GT_mse_array = []
    PERT_dice_array_SI_and_GT_rmse_array = []
    PERT_dice_array_SI_std_array = []
    SI_dice_array = []
    
    for image in tqdm(os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")):
        GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
        SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
        GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image
        SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
        MC_mask_path = general_results_path_2c + "_MC/" + subset + "/mask/" + image + "/" 
        PERT_mask_path = general_results_path_2c + "_perturbation/" + subset + "/mask/" + image + "/" 
        
        GT_mask_2c = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        if not GT_mask_2c.any(): continue
        # total_images += 1
        SI_mask_2c = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        SI_dice = wjb_functions.dice(GT_mask_2c,SI_mask_2c)
        SI_dice_array.append(SI_dice)
        str_dice = str(SI_dice)[:5]
        
        MC_path_2c = general_results_path_2c + "_MC/" + subset + "/" + "mask/" + image + "/"
        MC_softmax_path_2c = general_results_path_2c + "_MC/" + subset + "/" + "softmax/" + image + "/"
        
        PERT_path_2c = general_results_path_2c + "_perturbation/" + subset + "/" + "mask/" + image + "/"
        PERT_softmax_path_2c = general_results_path_2c + "_perturbation/" + subset + "/" + "softmax/" + image + "/"
        
        MC_softmax_matrix_2c = wjb_functions.softmax_matrix_gen(MC_softmax_path_2c, np.shape(GT_mask_2c)[:2], 2, N)
        MC_unc_map_2c = wjb_functions.binary_entropy_map(MC_softmax_matrix_2c[:,:,1,:])
        
        PERT_softmax_matrix_2c = wjb_functions.softmax_matrix_gen(PERT_softmax_path_2c, np.shape(GT_mask_2c)[:2], 2, N)
        PERT_unc_map_2c = wjb_functions.binary_entropy_map(PERT_softmax_matrix_2c[:,:,1,:])
        
        PERT_mask_avg = wjb_functions.mask_avg_gen_2c(PERT_softmax_matrix_2c)
        PERT_mask_union = wjb_functions.mask_union_gen_2c(PERT_path_2c)
        MC_mask_avg = wjb_functions.mask_avg_gen_2c(MC_softmax_matrix_2c)
        MC_mask_union = wjb_functions.mask_union_gen_2c(MC_path_2c)
        
        MC_dice_array_GT = []
        MC_dice_array_SI = []
        MC_dice_array_UM_MC = []
        MC_dice_array_avg_MC = []
        MC_dice_array_UM_PERT = []
        MC_dice_array_avg_PERT = []
        
        PERT_dice_array_GT = []
        PERT_dice_array_SI = []
        PERT_dice_array_UM_MC = []
        PERT_dice_array_avg_MC = []
        PERT_dice_array_UM_PERT = []
        PERT_dice_array_avg_PERT = []
        
        for n in os.listdir(PERT_mask_path):
            PERT_current_mask = cv2.imread(PERT_mask_path + n, cv2.IMREAD_GRAYSCALE).astype(bool)
            MC_current_mask = cv2.imread(MC_mask_path + n, cv2.IMREAD_GRAYSCALE).astype(bool)
            
            MC_dice_array_GT.append(wjb_functions.dice(MC_current_mask, GT_mask_2c))
            MC_dice_array_SI.append(wjb_functions.dice(MC_current_mask, SI_mask_2c))
            MC_dice_array_UM_MC.append(wjb_functions.dice(MC_current_mask, MC_mask_union))
            MC_dice_array_avg_MC.append(wjb_functions.dice(MC_current_mask, MC_mask_avg))
            MC_dice_array_UM_PERT.append(wjb_functions.dice(MC_current_mask, PERT_mask_union))
            MC_dice_array_avg_PERT.append(wjb_functions.dice(MC_current_mask, PERT_mask_avg))
            
            PERT_dice_array_GT.append(wjb_functions.dice(PERT_current_mask, GT_mask_2c))
            PERT_dice_array_SI.append(wjb_functions.dice(PERT_current_mask, SI_mask_2c))
            PERT_dice_array_UM_MC.append(wjb_functions.dice(PERT_current_mask, MC_mask_union))
            PERT_dice_array_avg_MC.append(wjb_functions.dice(PERT_current_mask, MC_mask_avg))
            PERT_dice_array_UM_PERT.append(wjb_functions.dice(PERT_current_mask, PERT_mask_union))
            PERT_dice_array_avg_PERT.append(wjb_functions.dice(PERT_current_mask, PERT_mask_avg))
                
        
        MC_dice_array_GT = np.array(MC_dice_array_GT)
        MC_dice_array_SI = np.array(MC_dice_array_SI)
        MC_dice_array_UM_MC = np.array(MC_dice_array_UM_MC)
        MC_dice_array_avg_MC = np.array(MC_dice_array_avg_MC)
        MC_dice_array_UM_PERT = np.array(MC_dice_array_UM_PERT)
        MC_dice_array_avg_PERT = np.array(MC_dice_array_avg_PERT)
        
        PERT_dice_array_GT = np.array(PERT_dice_array_GT)
        PERT_dice_array_SI = np.array(PERT_dice_array_SI)
        PERT_dice_array_UM_MC = np.array(PERT_dice_array_UM_MC)
        PERT_dice_array_avg_MC = np.array(PERT_dice_array_avg_MC)
        PERT_dice_array_UM_PERT = np.array(PERT_dice_array_UM_PERT)
        PERT_dice_array_avg_PERT = np.array(PERT_dice_array_avg_PERT)
        
        PERT_dice_array_SI_and_GT_mse_array.append(wjb_functions.mse(PERT_dice_array_SI,PERT_dice_array_GT))
        PERT_dice_array_SI_and_GT_rmse_array.append(wjb_functions.rmse(PERT_dice_array_SI,PERT_dice_array_GT))
        
        fig, axes = plt.subplots(2, 5, figsize=(50,10))
        fig.suptitle("Image " + image[:-4] + ", dataset " + dataset[:-1] + ", subset " + subset + ", SI DICE = " + str_dice).set_fontsize(40)
        
        # Define all the pairs of arrays and labels you want to plot
        plots = [
            (MC_dice_array_GT, "MC dice with Ground Truth", MC_dice_array_SI, "MC dice with Single Inference"),
            (MC_dice_array_GT, "MC dice with Ground Truth", MC_dice_array_UM_MC, "MC dice with Mask Union MC"),
            (MC_dice_array_GT, "MC dice with Ground Truth", MC_dice_array_avg_MC, "MC dice with Mask Avg MC"),
            (MC_dice_array_GT, "MC dice with Ground Truth", MC_dice_array_UM_PERT, "MC dice with Mask Union PERT"),
            (MC_dice_array_GT, "MC dice with Ground Truth", MC_dice_array_avg_PERT, "MC dice with Mask Avg PERT"),
            (PERT_dice_array_GT, "PERT dice with Ground Truth", PERT_dice_array_SI, "PERT dice with Single Inference"),
            (PERT_dice_array_GT, "PERT dice with Ground Truth", PERT_dice_array_UM_MC, "PERT dice with Mask Union MC"),
            (PERT_dice_array_GT, "PERT dice with Ground Truth", PERT_dice_array_avg_MC, "PERT dice with Mask Avg MC"),
            (PERT_dice_array_GT, "PERT dice with Ground Truth", PERT_dice_array_UM_PERT, "PERT dice with Mask Union PERT"),
            (PERT_dice_array_GT, "PERT dice with Ground Truth", PERT_dice_array_avg_PERT, "PERT dice with Mask Avg PERT")
        ]
        
        # Create subplots
        axes = []
        for i in range(2):
            for j in range(5):
                ax = plt.subplot(2, 5, i * 5 + j + 1)
                axes.append(ax)
        
        # Plot each subplot
        for ax, (y1, label1, y2, label2) in zip(axes, plots):
            ax.plot(y1, label=label1, color=colors_for_plot[3])
            ax.set_ylabel(label1, color=colors_for_plot[3])
            ax.tick_params(axis='y', labelcolor=colors_for_plot[3])
            
            ax2 = ax.twinx()
            ax2.plot(y2, label=label2, color=colors_for_plot[0])
            ax2.set_ylabel(label2, color=colors_for_plot[0])
            ax2.tick_params(axis='y', labelcolor=colors_for_plot[0])
        
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
            # Set x-axis ticks and labels to be from 1 to 20
            ax.set_xticks(range(0, 20))
            ax.set_xticklabels(range(0, 20))
        
        # Adjust layout to make room for the suptitle and spacing between subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1, wspace=0.4, hspace=0.4)
        
        if not os.path.isdir(general_path_to_save + dataset + "/" + subset): os.makedirs(general_path_to_save + dataset + "/" + subset)
        
        plt.savefig(general_path_to_save + dataset + "/" + subset + "/" + image[:-4] + "_dices_per_the_20_images.png")
        plt.close()
        
        PERT_dice_array_SI_std_array.append(np.nanstd(PERT_dice_array_SI))
        
    PERT_dice_array_SI_std_array = np.array(PERT_dice_array_SI_std_array)
    SI_dice_array = np.array(SI_dice_array)
    rho = wjb_functions.pearsons(PERT_dice_array_SI_std_array, SI_dice_array)[0][1]
    plt.figure(figsize=(12,12))
    plt.title("Dataset: " + dataset + ", subset: " + subset + ", correlation between std of PERT_dice_array_SI and SI_dice_array = " + str(rho)[:5])
    plt.scatter(PERT_dice_array_SI_std_array,SI_dice_array)
    plt.xlabel("PERT_dice_array_SI_std_array")
    plt.ylabel("SI_dice_array")
    plt.savefig(general_path_to_save + dataset + "/" + subset + "/000000_correlation_PERTdicearraySI_and_SIdicearray.png")
    plt.close()
    
    PERT_dice_array_SI_and_GT_mse_array = np.array(PERT_dice_array_SI_and_GT_mse_array)
    PERT_dice_array_SI_and_GT_rmse_array = np.array(PERT_dice_array_SI_and_GT_rmse_array)
    rho_mse = wjb_functions.pearsons(PERT_dice_array_SI_and_GT_mse_array, SI_dice_array)[0][1]
    rho_rmse = wjb_functions.pearsons(PERT_dice_array_SI_and_GT_rmse_array, SI_dice_array)[0][1]
    
    plt.figure(figsize=(12,12))
    plt.title("Dataset: " + dataset + ", subset: " + subset + ", correlation between mse of PERT_dice_array_SI_and_GT and SI_dice_array = " + str(rho_mse)[:5])
    plt.scatter(PERT_dice_array_SI_and_GT_mse_array,SI_dice_array)
    plt.xlabel("PERT_dice_array_SI_and_GT_mse_array")
    plt.ylabel("SI_dice_array")
    plt.savefig(general_path_to_save + dataset + "/" + subset + "/000000_correlation_msePERTdicearraySIandGT_and_SIdicearray.png")
    plt.close()
    
    plt.figure(figsize=(12,12))
    plt.title("Dataset: " + dataset + ", subset: " + subset + ", correlation between rmse of PERT_dice_array_SI_and_GT and SI_dice_array = " + str(rho_rmse)[:5])
    plt.scatter(PERT_dice_array_SI_and_GT_rmse_array,SI_dice_array)
    plt.xlabel("PERT_dice_array_SI_and_GT_rmse_array")
    plt.ylabel("SI_dice_array")
    plt.savefig(general_path_to_save + dataset + "/" + subset + "/000000_correlation_rmsePERTdicearraySIandGT_and_SIdicearray.png")
    plt.close()