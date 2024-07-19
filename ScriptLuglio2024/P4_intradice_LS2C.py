# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:50:29 2024

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

#%%

diff_type = "PERT"
diff_type_name = "_perturbation/"
dataset = "Liver HE steatosis/"
subset = "test"
image = "1004289_35.png"
N = 20
# radius = 5
c = 2

colors_for_plot = [                               
    to_rgb('#006400'),
    to_rgb('#bc8f8f'),
    to_rgb('#ffd700'),
    to_rgb('#0000cd'),
    to_rgb('#00ff00'),
    to_rgb('#00ffff'),
    to_rgb('#1e90ff'),
    to_rgb('#ff1493')]

#%%

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V25/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"


for subset in tqdm(["test", "val"]):
    # if subset == "val": continue
    SI_GT_dice_array = []
    mean_PERT_intradice_array = []
    mean_MC_intradice_array = []
    list_of_images = []
    
    for image in tqdm(os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")):
        
        path_to_save_figure = general_path_to_save + dataset + subset + "/"
        if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)

        GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
        SI_path_2c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
        mask_path_2c_MC = general_results_path_3c + "_MC" + "/" + subset + "/" + "mask/" + image + "/"
        softmax_path_2c_MC = general_results_path_3c + "_MC" + "/" + subset + "/" + "softmax/" + image + "/"
        mask_path_2c_PERT = general_results_path_3c + "_perturbation" + "/" + subset + "/" + "mask/" + image + "/"
        softmax_path_2c_PERT = general_results_path_3c + "_perturbation" + "/" + subset + "/" + "softmax/" + image + "/"
        
        
        #%%
        dataset_name = dataset[:-1] + "_3c/"
        GT_mask = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE)/255
        
        if not GT_mask.any(): continue
    
        list_of_images.append(image)
        
        SI_mask = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE)/255
        SI_GT_dice = wjb_functions.dice(SI_mask,GT_mask)
        SI_GT_dice_array.append(SI_GT_dice)
        
        DIM = np.shape(GT_mask)[:2]
        
        mask_matrix_PERT = wjb_functions.mask_matrix_gen_2c(mask_path_2c_PERT,DIM,N)
        mask_matrix_MC = wjb_functions.mask_matrix_gen_2c(mask_path_2c_MC,DIM,N)
        
        dice_mat_PERT = wjb_functions.intradice_mat(mask_matrix_PERT)
        dice_mat_MC = wjb_functions.intradice_mat(mask_matrix_MC)
        
        mean_PERT_intradice = np.nanmean(dice_mat_PERT)
        mean_PERT_intradice_array.append(mean_PERT_intradice)
        mean_MC_intradice = np.nanmean(dice_mat_MC)
        mean_MC_intradice_array.append(mean_MC_intradice)
        
        plt.figure(figsize=(12,5))
        plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4] + ", dice_SI = " + str(SI_GT_dice)[:4])
        plt.subplot(121)
        plt.title("PERT, mean dice = " + str(mean_PERT_intradice)[:4])
        plt.imshow(dice_mat_PERT)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(122)
        plt.title("MC, mean dice = " + str(mean_MC_intradice)[:4])
        plt.imshow(dice_mat_MC)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(path_to_save_figure+"intradice_maps_"+image)
        plt.close()
    
    mean_PERT_intradice_array = np.array(mean_PERT_intradice_array)
    mean_MC_intradice_array = np.array(mean_MC_intradice_array)
    SI_GT_dice_array = np.array(SI_GT_dice_array)
    
    rho_PERT = np.corrcoef(mean_PERT_intradice_array,SI_GT_dice_array)[0][1]
    rho_MC = np.corrcoef(mean_MC_intradice_array,SI_GT_dice_array)[0][1]
    
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.title("mean intradice pert x SI-GT dice, rho = " + str(rho_PERT)[:5])
    plt.scatter(mean_PERT_intradice_array,SI_GT_dice_array)
    plt.xlabel("mean perturbation intradice")
    plt.ylabel("single inference dice")
    plt.subplot(122)
    plt.title("mean intradice mc x SI-GT dice, rho = " + str(rho_MC)[:5])
    plt.scatter(mean_MC_intradice_array,SI_GT_dice_array)
    plt.xlabel("mean montecarlo intradice")
    plt.ylabel("single inference dice")
    plt.savefig(path_to_save_figure+"correlation_between_intradices_and_dices.png")
    plt.close()