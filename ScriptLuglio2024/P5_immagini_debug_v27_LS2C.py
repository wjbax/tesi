# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:54:45 2024

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

for dataset in ["Liver HE steatosis/"]:
    
    dataset_name = "Liver HE steatosis 2C/"
    
    # diff_type = "PERT"
    # diff_type_name = "_perturbation/"
    # dataset = "Renal PAS glomeruli/"
    # subset = "test"
    # image = "1004763_1.png"
    N = 20
    # radius = 5
    c = 2

    general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V27/"
    general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
    # general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
    # general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

    general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
    general_results_path_3c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
    
    for subset in tqdm(["test", "val"]):
        
        # image_path = os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")
        # GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
        # OR_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "image/" + image
        # OR_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "image/" + image
        # GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image

        image_path = os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")
        
        
        for image in tqdm(image_path):
            
            GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
            OR_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "image/" + image
            OR_path_3c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "image/" + image
            SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
            GT_path_3c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
            SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
            PERT_mask_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "mask/" + image + "/"
            PERT_softmax_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "softmax/" + image + "/"
            MC_mask_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "mask/" + image + "/"
            MC_softmax_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "softmax/" + image + "/"

            
            # diff_path_3c_MC = general_results_path_3c + "_MC/" + subset + "/" + "mask/" + image + "/"
            
            specific_path_to_save = general_path_to_save + dataset_name + subset + "/"
            if not os.path.isdir(specific_path_to_save): os.makedirs(specific_path_to_save)
            
            pts = specific_path_to_save + image[:-4] + ".png"
            
            #%%
            OR_image = cv2.imread(OR_path_3c)
            OR_image = cv2.cvtColor(OR_image, cv2.COLOR_BGR2RGB)
            GT_mask = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
            SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
            DIM = np.shape(GT_mask)[:2]
            
            C1_exist = True
            
            if not GT_mask.any(): C1_exist = False
            
            if not C1_exist: 
                print("Image " + image[:-4] + " has no GT")
                continue
            
            #%%
            
            SI_dice = wjb_functions.dice(SI_mask,GT_mask)
            
            mask_union_MC = wjb_functions.mask_union_gen_2c(MC_mask_path_3c)/255
            mask_union_PERT = wjb_functions.mask_union_gen_2c(PERT_mask_path_3c)/255
            
            
            GT_masked_image = mark_boundaries(OR_image, GT_mask.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
            SI_masked_image = mark_boundaries(OR_image, SI_mask.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)   
            best_mask_MC = np.zeros_like(GT_mask)
            best_mask_PERT = np.zeros_like(GT_mask)
                
            
                
                
            #%%
            softmax_matrix_PERT = wjb_functions.softmax_matrix_gen(PERT_softmax_path_3c, DIM, c, N)
            softmax_matrix_MC = wjb_functions.softmax_matrix_gen(MC_softmax_path_3c, DIM, c, N)

            #%%
            max_dice_PERT = 0
            dice_PERT = []
            TOT_dice = []
            for i in range(21):
                if i == 0: continue
                current_mask = cv2.imread(PERT_mask_path_3c + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
                dice_cm = wjb_functions.dice(current_mask,GT_mask)
                if dice_cm > max_dice_PERT:
                    # print(dice_cm_C1)
                    max_dice_PERT = dice_cm
                    best_mask_PERT = copy.deepcopy(current_mask)
                dice_PERT.append(dice_cm)
                TOT_dice.append(dice_cm)
        
            #%%
            
            max_dice_MC = 0
            dice_MC = []
            for i in range(20):
                if i == 0: continue
                current_mask = cv2.imread(MC_mask_path_3c + str(i) + ".png", cv2.IMREAD_GRAYSCALE)/255
                dice_cm = wjb_functions.dice(current_mask,GT_mask)
                if dice_cm > max_dice_MC:
                    # print(dice_cm)
                    max_dice_MC = dice_cm
                    best_mask_MC = copy.deepcopy(current_mask)
                dice_MC.append(dice_cm)
                TOT_dice.append(dice_cm)
                
            #%%
            
            PERT_masked_image = mark_boundaries(OR_image, best_mask_PERT.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
            MC_masked_image = mark_boundaries(OR_image, best_mask_MC.astype(int), color=(1,0,0), outline_color=(1,0,0), mode='outer', background_label=0)
            
            #%%
            MC_binent_map = wjb_functions.binary_entropy_map(softmax_matrix_MC[:,:,1,:])
            MC_BC_map = wjb_functions.mean_softmax_BC_2(softmax_matrix_MC)
            
            PERT_binent_map = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
            PERT_BC_map = wjb_functions.mean_softmax_BC_2(softmax_matrix_PERT)
            
            #%%
            
            BC_th_dice_array_MC = []
            BC_th_dice_array_PERT = []
            
            unc_map_MC = MC_BC_map
            unc_map_PERT = PERT_BC_map

            for th in th_range:
                mask_th_BC_MC = mask_union_MC.astype(bool) & (unc_map_MC<=th)
                BC_th_dice_array_MC.append(wjb_functions.dice(mask_th_BC_MC, GT_mask))
                mask_th_BC_PERT = mask_union_PERT.astype(bool) & (unc_map_PERT<=th)
                BC_th_dice_array_PERT.append(wjb_functions.dice(mask_th_BC_PERT, GT_mask))
            
            
            binent_th_dice_array_MC = []
            binent_th_dice_array_PERT = []
            
            unc_map_MC = MC_binent_map
            unc_map_PERT = PERT_binent_map

            for th in th_range:
                mask_th_binent_MC = mask_union_MC.astype(bool) & (unc_map_MC<=th)
                binent_th_dice_array_MC.append(wjb_functions.dice(mask_th_binent_MC, GT_mask))
                mask_th_binent_PERT = mask_union_PERT.astype(bool) & (unc_map_PERT<=th)
                binent_th_dice_array_PERT.append(wjb_functions.dice(mask_th_binent_PERT, GT_mask))
                
            binent_th_dice_array_MC = np.array(binent_th_dice_array_MC)
            binent_th_dice_array_PERT = np.array(binent_th_dice_array_PERT)
            BC_th_dice_array_MC = np.array(BC_th_dice_array_MC)
            BC_th_dice_array_PERT = np.array(BC_th_dice_array_PERT)
            
            
            #%%
            
            data = [SI_dice, dice_MC, dice_PERT, TOT_dice]
            
            x_label = ['SI', 'MC', 'Pert', 'TOT']
            
            #%%
            
            PERT_mask_matrix = wjb_functions.mask_matrix_gen_3c(PERT_mask_path_3c,DIM,N)/255            
            
            dice_mat_PERT = wjb_functions.intradice_mat(PERT_mask_matrix)
            mean_dice_mat_PERT = np.nanmean(dice_mat_PERT)
            
            MC_mask_matrix = wjb_functions.mask_matrix_gen_3c(MC_mask_path_3c,DIM,N)/255            
            
            dice_mat_MC = wjb_functions.intradice_mat(MC_mask_matrix)
            mean_dice_mat_MC = np.nanmean(dice_mat_MC)
           
            #%%
            
            plt.figure(figsize=(20,16))
            plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4]).set_fontsize(20)
            plt.subplot(341)
            plt.title("Ground Truth")
            plt.imshow(GT_masked_image)
            plt.subplot(342)
            plt.title("Single Inference, dice = " + str(SI_dice)[:5])
            plt.imshow(SI_masked_image)
            plt.subplot(343)
            plt.title("Boxplot dices pert and MC")
            plt.boxplot(data)
            plt.ylabel('DICE')
            plt.xticks([1,2,3,4], x_label)
            plt.subplot(345)
            plt.title("pert union mask dice x threshold")
            plt.plot(th_range,binent_th_dice_array_PERT, color = colors_for_plot[2], label="Pert - bin ent")
            plt.plot(th_range,BC_th_dice_array_PERT, color = colors_for_plot[6], label="Pert - BC")
            plt.axhline(SI_dice, linestyle='--', color='r')
            plt.xlabel("Threshold")
            plt.xlim(0,1)
            plt.ylabel("Dice union mask vs GT")
            plt.legend()
            plt.subplot(349)
            plt.title("mc union mask dice x threshold")
            plt.plot(th_range,binent_th_dice_array_MC, color = colors_for_plot[0], label="MC - bin ent")
            plt.plot(th_range,BC_th_dice_array_MC, color = colors_for_plot[4], label="MC - BC")
            plt.axhline(SI_dice, linestyle='--', color='r')
            plt.xlabel("Threshold")
            plt.xlim(0,1)
            plt.ylabel("Dice union mask vs GT")
            plt.legend()
            plt.subplot(3,4,10)
            plt.title("Bin Ent Uncertainty Map MC")
            plt.imshow(MC_binent_map)
            plt.colorbar()
            plt.subplot(3,4,11)
            plt.title("Bhattacharyya Unc Map MC")
            plt.imshow(MC_BC_map)
            plt.colorbar()
            plt.subplot(346)
            plt.title("Bin Ent Uncertainty Map PERT")
            plt.imshow(PERT_binent_map)
            plt.colorbar()
            plt.subplot(347)
            plt.title("Bhattacharyya Unc Map PERT")
            plt.imshow(PERT_BC_map)
            plt.colorbar()
            plt.subplot(3,4,12)
            plt.title("Intradice matrix MC, mean = " + str(mean_dice_mat_MC)[:5])
            plt.imshow(dice_mat_MC)
            plt.colorbar()
            plt.subplot(348)
            plt.title("Intradice matrix PERT, mean = " + str(mean_dice_mat_PERT)[:5])
            plt.imshow(dice_mat_PERT)
            plt.colorbar()
            plt.savefig(pts)
            plt.close()
        
            