# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:32:40 2024

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

# diff_type = "PERT"
# diff_type_name = "urbation/"
# dataset = "Liver HE steatosis/"
# subset = "test"
# image = "1004289_35.png"
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
for dataset in ["Renal PAS glomeruli/", "Renal PAS tubuli/"]:
    
    if dataset == "Renal PAS tubuli/": continue
    
    general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V25/"
    general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
    general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"
     
    for subset in tqdm(["test", "val"]):
        SI_GT_C1_dice_array = []
        SI_GT_C2_dice_array = []
        
        PERT_mean_C2_intradice_array = []
        PERT_mean_C1_intradice_array = []
        
        MC_mean_C2_intradice_array = []
        MC_mean_C1_intradice_array = []
        
        for image in tqdm(os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/")):
        # for image in os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/"):
            
            path_to_save_figure = general_path_to_save + dataset + subset + "/"
            if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
    
            GT_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
            SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
            PERT_mask_path_2c = general_results_path_3c + "_perturbation" + "/" + subset + "/" + "mask/" + image + "/"
            PERT_softmax_path_2c = general_results_path_3c + "_perturbation" + "/" + subset + "/" + "softmax/" + image + "/"
            MC_mask_path_2c = general_results_path_3c + "_MC" + "/" + subset + "/" + "mask/" + image + "/"
            MC_softmax_path_2c = general_results_path_3c + "_MC" + "/" + subset + "/" + "softmax/" + image + "/"
            
            
            GT_mask_original = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
            GT_mask_C1,GT_mask_C2 = wjb_functions.mask_splitter(GT_mask_original)
            
            SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
            SI_mask_C1,SI_mask_C2 = wjb_functions.mask_splitter(SI_mask)
            
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(GT_mask_original)
            # plt.subplot(122)
            # plt.imshow(SI_mask)
            # plt.show()
            
            DIM = np.shape(GT_mask_original)[:2]
            
            C1_exist = True
            C2_exist = True
            if not GT_mask_C1.any(): 
                C1_exist = False
                # print("Image " + image[:-4] + " has no GT C1")
            if not GT_mask_C2.any(): 
                C2_exist = False
                # print("Image " + image[:-4] + " has no GT C2")
            
            if not C1_exist and not C2_exist: 
                print("Image " + image[:-4] + " has no GT")
                continue
            
            if C1_exist and C2_exist:
                # print("Image " + image[:-4] + ", C1 and C2 exist")
                SI_GT_C1_dice = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
                SI_GT_C2_dice = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
                SI_GT_C1_dice_array.append(SI_GT_C1_dice)
                SI_GT_C2_dice_array.append(SI_GT_C2_dice)
                
                PERT_mask_matrix = wjb_functions.mask_matrix_gen_3c(PERT_mask_path_2c,DIM,N)/255            
                PERT_mask_matrix_C1,PERT_mask_matrix_C2 = wjb_functions.mask_splitter(PERT_mask_matrix)
                
                PERT_dice_mat_C1 = wjb_functions.intradice_mat(PERT_mask_matrix_C1)
                
                PERT_mean_C1_intradice = np.nanmean(PERT_dice_mat_C1)
                PERT_mean_C1_intradice_array.append(PERT_mean_C1_intradice)
                
                PERT_dice_mat_C2 = wjb_functions.intradice_mat(PERT_mask_matrix_C2)
                
                PERT_mean_C2_intradice = np.nanmean(PERT_dice_mat_C2)
                PERT_mean_C2_intradice_array.append(PERT_mean_C2_intradice)
                
                
                MC_mask_matrix = wjb_functions.mask_matrix_gen_3c(MC_mask_path_2c,DIM,N)/255            
                MC_mask_matrix_C1,MC_mask_matrix_C2 = wjb_functions.mask_splitter(MC_mask_matrix)
                
                MC_dice_mat_C1 = wjb_functions.intradice_mat(MC_mask_matrix_C1)
                
                MC_mean_C1_intradice = np.nanmean(MC_dice_mat_C1)
                MC_mean_C1_intradice_array.append(MC_mean_C1_intradice)
                
                MC_dice_mat_C2 = wjb_functions.intradice_mat(MC_mask_matrix_C2)
                
                MC_mean_C2_intradice = np.nanmean(MC_dice_mat_C2)
                MC_mean_C2_intradice_array.append(MC_mean_C2_intradice)
                
                
                
                plt.figure(figsize=(12,10))
                plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4])
                plt.subplot(221)
                plt.title("PERT - C1, mean dice = " + str(PERT_mean_C1_intradice)[:5] + ", SI dice = " + str(SI_GT_C1_dice)[:5])
                plt.imshow(PERT_dice_mat_C1)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(222)
                plt.title("PERT - C2, mean dice = " + str(PERT_mean_C2_intradice)[:5] + ", SI dice = " + str(SI_GT_C2_dice)[:5])
                plt.imshow(PERT_dice_mat_C2)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(223)
                plt.title("MC - C1, mean dice = " + str(MC_mean_C1_intradice)[:5] + ", SI dice = " + str(SI_GT_C1_dice)[:5])
                plt.imshow(MC_dice_mat_C1)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(224)
                plt.title("MC - C2, mean dice = " + str(MC_mean_C2_intradice)[:5] + ", SI dice = " + str(SI_GT_C2_dice)[:5])
                plt.imshow(MC_dice_mat_C2)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.savefig(path_to_save_figure+"intradice_maps_"+image)
                plt.close()
            
            if C1_exist and not C2_exist:
                # print("Image " + image[:-4] + ", C1 exist and C2 does not exist")
                SI_GT_C1_dice_array.append(wjb_functions.dice(SI_mask_C1,GT_mask_C1))
                SI_GT_C1_dice = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
                
                PERT_mask_matrix = wjb_functions.mask_matrix_gen_3c(PERT_mask_path_2c,DIM,N)/255            
                PERT_mask_matrix_C1,PERT_mask_matrix_C2 = wjb_functions.mask_splitter(PERT_mask_matrix)
                
                PERT_dice_mat_C1 = wjb_functions.intradice_mat(PERT_mask_matrix_C1)
                
                PERT_mean_C1_intradice = np.nanmean(PERT_dice_mat_C1)
                PERT_mean_C1_intradice_array.append(PERT_mean_C1_intradice)
                
                
                MC_mask_matrix = wjb_functions.mask_matrix_gen_3c(MC_mask_path_2c,DIM,N)/255            
                MC_mask_matrix_C1,MC_mask_matrix_C2 = wjb_functions.mask_splitter(MC_mask_matrix)
                
                MC_dice_mat_C1 = wjb_functions.intradice_mat(MC_mask_matrix_C1)
                
                MC_mean_C1_intradice = np.nanmean(MC_dice_mat_C1)
                MC_mean_C1_intradice_array.append(MC_mean_C1_intradice)
                
                plt.figure(figsize=(6,11))
                plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4] + " - (no C2)")
                plt.subplot(211)
                plt.title("PERT - C1, mean dice = " + str(PERT_mean_C1_intradice)[:5] + ", SI dice = " + str(SI_GT_C1_dice)[:5])
                plt.imshow(PERT_dice_mat_C1)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(212)
                plt.title("MC - C1, mean dice = " + str(MC_mean_C1_intradice)[:5] + ", SI dice = " + str(SI_GT_C1_dice)[:5])
                plt.imshow(MC_dice_mat_C1)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.savefig(path_to_save_figure+"intradice_maps_"+image)
                plt.close()
                
            if C2_exist and not C1_exist:
                # print("Image " + image[:-4] + ", C1 exist and C2 does not exist")
                SI_GT_C2_dice_array.append(wjb_functions.dice(SI_mask_C2,GT_mask_C2))
                SI_GT_C2_dice = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
                
                PERT_mask_matrix = wjb_functions.mask_matrix_gen_3c(PERT_mask_path_2c,DIM,N)/255            
                PERT_mask_matrix_C2,PERT_mask_matrix_C2 = wjb_functions.mask_splitter(PERT_mask_matrix)
                
                PERT_dice_mat_C2 = wjb_functions.intradice_mat(PERT_mask_matrix_C2)
                
                PERT_mean_C2_intradice = np.nanmean(PERT_dice_mat_C2)
                PERT_mean_C2_intradice_array.append(PERT_mean_C2_intradice)
                
                
                MC_mask_matrix = wjb_functions.mask_matrix_gen_3c(MC_mask_path_2c,DIM,N)/255            
                MC_mask_matrix_C2,MC_mask_matrix_C2 = wjb_functions.mask_splitter(MC_mask_matrix)
                
                MC_dice_mat_C2 = wjb_functions.intradice_mat(MC_mask_matrix_C2)
                
                MC_mean_C2_intradice = np.nanmean(MC_dice_mat_C2)
                MC_mean_C2_intradice_array.append(MC_mean_C2_intradice)
                
                plt.figure(figsize=(6,11))
                plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4] + " - (no C1)")
                plt.subplot(211)
                plt.title("PERT - C2, mean dice = " + str(PERT_mean_C2_intradice)[:5] + ", SI dice = " + str(SI_GT_C2_dice)[:5])
                plt.imshow(PERT_dice_mat_C2)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.subplot(212)
                plt.title("PERT - C2, mean dice = " + str(MC_mean_C2_intradice)[:5] + ", SI dice = " + str(SI_GT_C2_dice)[:5])
                plt.imshow(MC_dice_mat_C2)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.savefig(path_to_save_figure+"intradice_maps_"+image)
                plt.close()
            
        PERT_mean_C1_intradice_array = np.array(PERT_mean_C1_intradice_array)
        SI_GT_C1_dice_array = np.array(SI_GT_C1_dice_array)
        
        PERT_rho_C1 = wjb_functions.pearsons(PERT_mean_C1_intradice_array,SI_GT_C1_dice_array)[0][1]
        
        PERT_mean_C2_intradice_array = np.array(PERT_mean_C2_intradice_array)
        SI_GT_C2_dice_array = np.array(SI_GT_C2_dice_array)
        
        PERT_rho_C2 = wjb_functions.pearsons(PERT_mean_C2_intradice_array,SI_GT_C2_dice_array)[0][1]
        
        MC_mean_C1_intradice_array = np.array(MC_mean_C1_intradice_array)
        
        MC_rho_C1 = wjb_functions.pearsons(MC_mean_C1_intradice_array,SI_GT_C1_dice_array)[0][1]
        
        MC_mean_C2_intradice_array = np.array(MC_mean_C2_intradice_array)
        
        MC_rho_C2 = wjb_functions.pearsons(MC_mean_C2_intradice_array,SI_GT_C2_dice_array)[0][1]
        
        
        plt.figure(figsize=(16,15))
        plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset)
        plt.subplot(221)
        plt.title("PERT - C1: mean intradice pert x SI-GT dice, rho = " + str(PERT_rho_C1)[:5])
        plt.scatter(PERT_mean_C1_intradice_array,SI_GT_C1_dice_array)
        plt.xlabel("mean perturbation intradice C1")
        plt.ylabel("single inference dice")
        plt.subplot(222)
        plt.title("PERT - C2: mean intradice pert x SI-GT dice, rho = " + str(PERT_rho_C2)[:5])
        plt.scatter(PERT_mean_C2_intradice_array,SI_GT_C2_dice_array)
        plt.xlabel("mean perturbation intradice C2")
        plt.ylabel("single inference dice")
        plt.subplot(223)
        plt.title("MC - C1: mean intradice MC x SI-GT dice, rho = " + str(MC_rho_C1)[:5])
        plt.scatter(MC_mean_C1_intradice_array,SI_GT_C1_dice_array)
        plt.xlabel("mean MCurbation intradice C1")
        plt.ylabel("single inference dice")
        plt.subplot(224)
        plt.title("MC - C2: mean intradice MC x SI-GT dice, rho = " + str(MC_rho_C2)[:5])
        plt.scatter(MC_mean_C2_intradice_array,SI_GT_C2_dice_array)
        plt.xlabel("mean MCurbation intradice C2")
        plt.ylabel("single inference dice")
        plt.savefig(path_to_save_figure+"correlation_between_intradices_and_dices.png")
        plt.close()