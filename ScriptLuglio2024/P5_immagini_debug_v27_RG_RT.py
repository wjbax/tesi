# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 02:17:17 2024

@author: willy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import cv2
import os
from skimage import measure
from skimage.segmentation import mark_boundaries
import copy
from tqdm import tqdm
import wjb_functions
import pandas as pd

#%%
colors_for_plot = [                               
    to_rgb('#006400'),
    to_rgb('#bc8f8f'),
    to_rgb('#ffd700'),
    to_rgb('#0000cd'),
    to_rgb('#00ff00'),
    to_rgb('#00ffff'),
    to_rgb('#1e90ff'),
    to_rgb('#ff1493')]

th_range = np.arange(1,11,1)/10

for dataset in ["Renal PAS glomeruli/", "Renal PAS tubuli/"]:
    
    # diff_type = "PERT"
    # diff_type_name = "_perturbation/"
    # dataset = "Renal PAS glomeruli/"
    # subset = "test"
    # image = "1004763_1.png"
    N = 20
    # radius = 5
    c = 3

    general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V27/"
    general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
    # general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
    # general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

    general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    
    for subset in tqdm(["test", "val"]):
        
        # image_path = os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")
        # GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
        # OR_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "image/" + image
        # OR_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "image/" + image
        # GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image

        image_path = os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/")
        
        
        for image in tqdm(image_path):
            
            GT_path_2c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
            OR_path_2c = general_dataset_path + "DATASET/" + subset + "/" + "image/" + image
            OR_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "image/" + image
            SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
            GT_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
            SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
            PERT_mask_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "mask/" + image + "/"
            PERT_softmax_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "softmax/" + image + "/"
            MC_mask_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "mask/" + image + "/"
            MC_softmax_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "softmax/" + image + "/"

            
            # diff_path_3c_MC = general_results_path_3c + "_MC/" + subset + "/" + "mask/" + image + "/"
            
            specific_path_to_save = general_path_to_save + dataset + subset + "/"
            if not os.path.isdir(specific_path_to_save): os.makedirs(specific_path_to_save)
            
            pts_C1 = specific_path_to_save + image[:-4] + "_C1.png"
            pts_C2 = specific_path_to_save + image[:-4] + "_C2.png"
            
            #%%
            OR_image = cv2.imread(OR_path_3c)
            OR_image = cv2.cvtColor(OR_image, cv2.COLOR_BGR2RGB)
            GT_mask = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)
            GT_mask_C1,GT_mask_C2 = wjb_functions.mask_splitter(GT_mask/255)
            SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)
            SI_mask_C1,SI_mask_C2 = wjb_functions.mask_splitter(SI_mask/255)
            DIM = np.shape(GT_mask)[:2]
            
            C1_exist = True
            C2_exist = True
            if not GT_mask_C1.any(): C1_exist = False
            if not GT_mask_C2.any(): C2_exist = False
            
            if not C1_exist and not C2_exist: 
                print("Image " + image[:-4] + " has no GT")
                continue
            
            #%%
            
            SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
            SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
            
            mask_union_MC = wjb_functions.mask_union_gen_3c(MC_mask_path_3c)/255
            mask_union_C1_MC,mask_union_C2_MC = wjb_functions.mask_splitter(mask_union_MC)
            mask_union_PERT = wjb_functions.mask_union_gen_3c(PERT_mask_path_3c)/255
            mask_union_C1_PERT,mask_union_C2_PERT = wjb_functions.mask_splitter(mask_union_PERT)
            
            if C1_exist:
                GT_masked_image_C1 = mark_boundaries(OR_image, GT_mask_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
                SI_masked_image_C1 = mark_boundaries(OR_image, SI_mask_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)   
                best_mask_MC_C1 = np.zeros_like(GT_mask_C1)
                best_mask_PERT_C1 = np.zeros_like(GT_mask_C1)
                
                
                
            if C2_exist:
                SI_masked_image_C2 = mark_boundaries(OR_image, SI_mask_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
                GT_masked_image_C2 = mark_boundaries(OR_image, GT_mask_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
                best_mask_MC_C2 = np.zeros_like(GT_mask_C2)
                best_mask_PERT_C2 = np.zeros_like(GT_mask_C2)
                
                
                
            #%%
            softmax_matrix_PERT = wjb_functions.softmax_matrix_gen(PERT_softmax_path_3c, DIM, c, N)
            softmax_matrix_MC = wjb_functions.softmax_matrix_gen(MC_softmax_path_3c, DIM, c, N)

            #%%
            max_dice_PERT_C2 = 0
            max_dice_PERT_C1 = 0
            dice_PERT_C1 = []
            dice_PERT_C2 = []
            TOT_dice_C1 = []
            TOT_dice_C2 = []
            for i in range(21):
                if i == 0: continue
                current_mask = cv2.imread(PERT_mask_path_3c + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
                current_mask_C1,current_mask_C2 = wjb_functions.mask_splitter(current_mask)
                if C2_exist:
                    dice_cm_C2 = wjb_functions.dice(current_mask_C2,GT_mask_C2)
                    if dice_cm_C2 > max_dice_PERT_C2:
                        # print(dice_cm_C2)
                        max_dice_PERT_C2 = dice_cm_C2
                        best_mask_PERT_C2 = copy.deepcopy(current_mask_C2)
                    dice_PERT_C2.append(dice_cm_C2)
                    TOT_dice_C2.append(dice_cm_C2)
                if C1_exist:
                    dice_cm_C1 = wjb_functions.dice(current_mask_C1,GT_mask_C1)
                    if dice_cm_C1 > max_dice_PERT_C1:
                        # print(dice_cm_C1)
                        max_dice_PERT_C1 = dice_cm_C1
                        best_mask_PERT_C1 = copy.deepcopy(current_mask_C1)
                    dice_PERT_C1.append(dice_cm_C1)
                    TOT_dice_C1.append(dice_cm_C1)
            
            #%%
            
            max_dice_MC_C1 = 0
            max_dice_MC_C2 = 0
            dice_MC_C1 = []
            dice_MC_C2 = []
            for i in range(20):
                if i == 0: continue
                current_mask = cv2.imread(MC_mask_path_3c + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
                current_mask_C1,current_mask_C2 = wjb_functions.mask_splitter(current_mask)
                if C2_exist:
                    dice_cm_C2 = wjb_functions.dice(current_mask_C2,GT_mask_C2)
                    if dice_cm_C2 > max_dice_MC_C2:
                        # print(dice_cm_C2)
                        max_dice_MC_C2 = dice_cm_C2
                        best_mask_MC_C2 = copy.deepcopy(current_mask_C2)
                    dice_MC_C2.append(dice_cm_C2)
                    TOT_dice_C2.append(dice_cm_C2)
                if C1_exist:
                    dice_cm_C1 = wjb_functions.dice(current_mask_C1,GT_mask_C1)
                    if dice_cm_C1 > max_dice_MC_C1:
                        # print(dice_cm_C1)
                        max_dice_MC_C1 = dice_cm_C1
                        best_mask_MC_C1 = copy.deepcopy(current_mask_C1)
                    dice_MC_C1.append(dice_cm_C1)
                    TOT_dice_C1.append(dice_cm_C1)
                
            #%%
            if C1_exist:
                PERT_masked_image_C1 = mark_boundaries(OR_image, best_mask_PERT_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
                MC_masked_image_C1 = mark_boundaries(OR_image, best_mask_MC_C1.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
            if C2_exist:
                PERT_masked_image_C2 = mark_boundaries(OR_image, best_mask_PERT_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
                MC_masked_image_C2 = mark_boundaries(OR_image, best_mask_MC_C2.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
            
            #%%
            MC_binent_map_C1 = wjb_functions.binary_entropy_map(softmax_matrix_MC[:,:,1,:])
            MC_binent_map_C2 = wjb_functions.binary_entropy_map(softmax_matrix_MC[:,:,2,:])
            MC_BC_map = wjb_functions.mean_softmax_BC_3(softmax_matrix_MC)
            
            PERT_binent_map_C1 = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
            PERT_binent_map_C2 = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,2,:])
            PERT_BC_map = wjb_functions.mean_softmax_BC_3(softmax_matrix_PERT)
            
            #%%
            
            BC_th_dice_array_C1_MC = []
            BC_th_dice_array_C2_MC = []
            BC_th_dice_array_C1_PERT = []
            BC_th_dice_array_C2_PERT = []
            
            unc_map_C1_MC = MC_BC_map
            unc_map_C2_MC = MC_BC_map
            unc_map_C1_PERT = PERT_BC_map
            unc_map_C2_PERT = PERT_BC_map

            for th in th_range:
                mask_th_BC_C1_MC = mask_union_C1_MC.astype(bool) & (unc_map_C1_MC<=th)
                BC_th_dice_array_C1_MC.append(wjb_functions.dice(mask_th_BC_C1_MC, GT_mask_C1))
                mask_th_BC_C2_MC = mask_union_C2_MC.astype(bool) & (unc_map_C2_MC<=th)
                BC_th_dice_array_C2_MC.append(wjb_functions.dice(mask_th_BC_C2_MC, GT_mask_C2))
                mask_th_BC_C1_PERT = mask_union_C1_PERT.astype(bool) & (unc_map_C1_PERT<=th)
                BC_th_dice_array_C1_PERT.append(wjb_functions.dice(mask_th_BC_C1_PERT, GT_mask_C1))
                mask_th_BC_C2_PERT = mask_union_C2_PERT.astype(bool) & (unc_map_C2_PERT<=th)
                BC_th_dice_array_C2_PERT.append(wjb_functions.dice(mask_th_BC_C2_PERT, GT_mask_C2))
            
            
            binent_th_dice_array_C1_MC = []
            binent_th_dice_array_C2_MC = []
            binent_th_dice_array_C1_PERT = []
            binent_th_dice_array_C2_PERT = []
            
            unc_map_C1_MC = MC_binent_map_C1
            unc_map_C2_MC = MC_binent_map_C2
            unc_map_C1_PERT = PERT_binent_map_C1
            unc_map_C2_PERT = PERT_binent_map_C2

            for th in th_range:
                mask_th_binent_C1_MC = mask_union_C1_MC.astype(bool) & (unc_map_C1_MC<=th)
                binent_th_dice_array_C1_MC.append(wjb_functions.dice(mask_th_binent_C1_MC, GT_mask_C1))
                mask_th_binent_C2_MC = mask_union_C2_MC.astype(bool) & (unc_map_C2_MC<=th)
                binent_th_dice_array_C2_MC.append(wjb_functions.dice(mask_th_binent_C2_MC, GT_mask_C2))
                mask_th_binent_C1_PERT = mask_union_C1_PERT.astype(bool) & (unc_map_C1_PERT<=th)
                binent_th_dice_array_C1_PERT.append(wjb_functions.dice(mask_th_binent_C1_PERT, GT_mask_C1))
                mask_th_binent_C2_PERT = mask_union_C2_PERT.astype(bool) & (unc_map_C2_PERT<=th)
                binent_th_dice_array_C2_PERT.append(wjb_functions.dice(mask_th_binent_C2_PERT, GT_mask_C2))
                
            binent_th_dice_array_C1_MC = np.array(binent_th_dice_array_C1_MC)
            binent_th_dice_array_C2_MC = np.array(binent_th_dice_array_C2_MC)
            binent_th_dice_array_C1_PERT = np.array(binent_th_dice_array_C1_PERT)
            binent_th_dice_array_C2_PERT = np.array(binent_th_dice_array_C2_PERT)
            BC_th_dice_array_C1_MC = np.array(BC_th_dice_array_C1_MC)
            BC_th_dice_array_C2_MC = np.array(BC_th_dice_array_C2_MC)
            BC_th_dice_array_C1_PERT = np.array(BC_th_dice_array_C1_PERT)
            BC_th_dice_array_C2_PERT = np.array(BC_th_dice_array_C2_PERT)
            
            
            #%%
            
            data_C1 = [SI_dice_C1, dice_MC_C1, dice_PERT_C1, TOT_dice_C1]
            data_C2 = [SI_dice_C2, dice_MC_C2, dice_PERT_C2, TOT_dice_C2]
            
            x_label_C1 = ['SI', 'MC', 'Pert', 'TOT']
            x_label_C2 = ['SI', 'MC', 'Pert', 'TOT']
            
            #%%
            
            PERT_mask_matrix = wjb_functions.mask_matrix_gen_3c(PERT_mask_path_3c,DIM,N)/255            
            PERT_mask_matrix_C1,PERT_mask_matrix_C2 = wjb_functions.mask_splitter(PERT_mask_matrix)
            
            if C1_exist:
                dice_mat_PERT_C1 = wjb_functions.intradice_mat(PERT_mask_matrix_C1)
                mean_dice_mat_PERT_C1 = np.nanmean(dice_mat_PERT_C1)
            if C2_exist:
                dice_mat_PERT_C2 = wjb_functions.intradice_mat(PERT_mask_matrix_C2)
                mean_dice_mat_PERT_C2 = np.nanmean(dice_mat_PERT_C2)
            
            MC_mask_matrix = wjb_functions.mask_matrix_gen_3c(MC_mask_path_3c,DIM,N)/255            
            MC_mask_matrix_C1,MC_mask_matrix_C2 = wjb_functions.mask_splitter(MC_mask_matrix)
            
            if C1_exist:
                dice_mat_MC_C1 = wjb_functions.intradice_mat(MC_mask_matrix_C1)
                mean_dice_mat_MC_C1 = np.nanmean(dice_mat_MC_C1)
            if C2_exist:
                dice_mat_MC_C2 = wjb_functions.intradice_mat(MC_mask_matrix_C2)
                mean_dice_mat_MC_C2 = np.nanmean(dice_mat_MC_C2)
            
            #%%
            if C1_exist:
                plt.figure(figsize=(20,16))
                plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4] + ", class 1").set_fontsize(20)
                plt.subplot(341)
                plt.title("Ground Truth")
                plt.imshow(GT_masked_image_C1)
                plt.subplot(342)
                plt.title("Single Inference, dice = " + str(SI_dice_C1)[:5])
                plt.imshow(SI_masked_image_C1)
                plt.subplot(343)
                plt.title("Boxplot dices pert and MC")
                plt.boxplot(data_C1)
                plt.ylabel('DICE')
                plt.xticks([1,2,3,4], x_label_C1)
                plt.subplot(345)
                plt.title("pert union mask dice x threshold")
                plt.plot(th_range,binent_th_dice_array_C1_PERT, color = colors_for_plot[2], label="Pert - bin ent")
                plt.plot(th_range,BC_th_dice_array_C1_PERT, color = colors_for_plot[6], label="Pert - BC")
                plt.axhline(SI_dice_C1, linestyle='--', color='r')
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.ylabel("Dice union mask vs GT")
                plt.legend()
                plt.subplot(349)
                plt.title("mc union mask dice x threshold")
                plt.plot(th_range,binent_th_dice_array_C1_MC, color = colors_for_plot[0], label="MC - bin ent")
                plt.plot(th_range,BC_th_dice_array_C1_MC, color = colors_for_plot[4], label="MC - BC")
                plt.axhline(SI_dice_C1, linestyle='--', color='r')
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.ylabel("Dice union mask vs GT")
                plt.legend()
                plt.subplot(3,4,10)
                plt.title("Bin Ent Uncertainty Map MC")
                plt.imshow(MC_binent_map_C1)
                plt.colorbar()
                plt.subplot(3,4,11)
                plt.title("Bhattacharyya Unc Map MC")
                plt.imshow(MC_BC_map)
                plt.colorbar()
                plt.subplot(346)
                plt.title("Bin Ent Uncertainty Map PERT")
                plt.imshow(PERT_binent_map_C1)
                plt.colorbar()
                plt.subplot(347)
                plt.title("Bhattacharyya Unc Map PERT")
                plt.imshow(PERT_BC_map)
                plt.colorbar()
                plt.subplot(3,4,12)
                plt.title("Intradice matrix MC, mean = " + str(mean_dice_mat_MC_C1)[:5])
                plt.imshow(dice_mat_MC_C1)
                plt.colorbar()
                plt.subplot(348)
                plt.title("Intradice matrix PERT, mean = " + str(mean_dice_mat_PERT_C1)[:5])
                plt.imshow(dice_mat_PERT_C1)
                plt.colorbar()
                plt.savefig(pts_C1)
                plt.close()
            
            if C2_exist:
                plt.figure(figsize=(20,16))
                plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4] + ", class 2").set_fontsize(20)
                plt.subplot(341)
                plt.title("Ground Truth")
                plt.imshow(GT_masked_image_C2)
                plt.subplot(342)
                plt.title("Single Inference, dice = " + str(SI_dice_C2)[:5])
                plt.imshow(SI_masked_image_C2)
                plt.subplot(343)
                plt.title("Boxplot dices pert and MC")
                plt.boxplot(data_C2)
                plt.ylabel('DICE')
                plt.xticks([1,2,3,4], x_label_C2)
                plt.subplot(345)
                plt.title("pert union mask dice x threshold")
                plt.plot(th_range,binent_th_dice_array_C2_PERT, color = colors_for_plot[3], label="Pert - bin ent")
                plt.plot(th_range,BC_th_dice_array_C2_PERT, color = colors_for_plot[7], label="Pert - BC")
                plt.axhline(SI_dice_C2, linestyle='--', color='r')
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.ylabel("Dice union mask vs GT")
                plt.legend()
                plt.subplot(349)
                plt.title("mc union mask dice x threshold")
                plt.plot(th_range,binent_th_dice_array_C2_MC, color = colors_for_plot[3], label="MC - bin ent")
                plt.plot(th_range,BC_th_dice_array_C2_MC, color = colors_for_plot[7], label="MC - BC")
                plt.axhline(SI_dice_C2, linestyle='--', color='r')
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.ylabel("Dice union mask vs GT")
                plt.legend()
                plt.subplot(3,4,10)
                plt.title("Bin Ent Uncertainty Map MC")
                plt.imshow(MC_binent_map_C2)
                plt.colorbar()
                plt.subplot(3,4,11)
                plt.title("Bhattacharyya Unc Map MC")
                plt.imshow(MC_BC_map)
                plt.colorbar()
                plt.subplot(346)
                plt.title("Bin Ent Uncertainty Map PERT")
                plt.imshow(PERT_binent_map_C2)
                plt.colorbar()
                plt.subplot(347)
                plt.title("Bhattacharyya Unc Map PERT")
                plt.imshow(PERT_BC_map)
                plt.colorbar()
                plt.subplot(3,4,12)
                plt.title("Intradice matrix MC, mean = " + str(mean_dice_mat_MC_C2)[:5])
                plt.imshow(dice_mat_MC_C2)
                plt.colorbar()
                plt.subplot(348)
                plt.title("Intradice matrix PERT, mean = " + str(mean_dice_mat_PERT_C2)[:5])
                plt.imshow(dice_mat_PERT_C2)
                plt.colorbar()
                plt.savefig(pts_C2)
                plt.close()