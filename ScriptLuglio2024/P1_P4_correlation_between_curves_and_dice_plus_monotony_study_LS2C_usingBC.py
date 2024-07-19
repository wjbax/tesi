# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:13:56 2024

@author: willy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:43:06 2024

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
dataset = "Liver HE steatosis/"
# subset = "val"
# image = "1004289_35.png"
N = 20
radius = 5

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V18_V2BC/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset

better_dices = 0
total_images = 0
for subset in tqdm(["test", "val"]):
    rho_tra_le_curve_MC = []
    rho_tra_le_curve_PERT = []
    rho_tra_le_curve_y = []
    eucl_dist_MC = []
    eucl_dist_PERT = []
    eucl_dist_y = []
    SI_dice_array = []
    monotony_corr_avg_y = []
    monotony_corr_SI_y = []
    monotony_corr_avg_MC = []
    monotony_corr_SI_MC = []
    monotony_corr_avg_PERT = []
    monotony_corr_SI_PERT = []
    
    for image in tqdm(os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")):
        
        general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
        general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"
        GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
        SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
        GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image
        SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
        
        _,GT_mask_3c_int = wjb_functions.mask_splitter(cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE))
        _,SI_mask_3c_int = wjb_functions.mask_splitter(cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE))
        GT_mask_2c = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        if not GT_mask_2c.any(): continue
        total_images += 1
        SI_mask_2c = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        MC_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "mask/" + image + "/"
        MC_softmax_path_2c = general_results_path_2c + "_MC/" + subset + "/" + "softmax/" + image + "/"
        MC_softmax_path_3c = general_results_path_3c + "_MC/" + subset + "/" + "softmax/" + image + "/"
        
        PERT_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "mask/" + image + "/"
        PERT_softmax_path_2c = general_results_path_2c + "_perturbation/" + subset + "/" + "softmax/" + image + "/"
        PERT_softmax_path_3c = general_results_path_3c + "_perturbation/" + subset + "/" + "softmax/" + image + "/"
        
        MC_mask_union_3c_int = wjb_functions.mask_union_gen(MC_path_3c)
        MC_softmax_matrix_2c = wjb_functions.softmax_matrix_gen(MC_softmax_path_2c, np.shape(GT_mask_2c)[:2], 2, N)
        MC_softmax_matrix_3c = wjb_functions.softmax_matrix_gen(MC_softmax_path_3c, np.shape(GT_mask_2c)[:2], 3, N)
        MC_mask_avg_3c_int = wjb_functions.mask_avg_gen(MC_softmax_matrix_3c)
        MC_unc_map_3c = wjb_functions.mean_softmax_BC_3(MC_softmax_matrix_3c)
        
        PERT_mask_union_3c_int = wjb_functions.mask_union_gen(PERT_path_3c)
        PERT_softmax_matrix_2c = wjb_functions.softmax_matrix_gen(PERT_softmax_path_2c, np.shape(GT_mask_2c)[:2], 2, N)
        PERT_softmax_matrix_3c = wjb_functions.softmax_matrix_gen(PERT_softmax_path_3c, np.shape(GT_mask_2c)[:2], 3, N)
        PERT_mask_avg_3c_int = wjb_functions.mask_avg_gen(PERT_softmax_matrix_3c)
        PERT_unc_map_3c = wjb_functions.mean_softmax_BC_3(PERT_softmax_matrix_3c)
        
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
        MC_mask_avg_dice_with_SI = wjb_functions.dice(MC_mask_avg_3c_int,SI_mask_2c)
        MC_mask_union_dice_with_avg = wjb_functions.dice(MC_mask_union_3c_int,MC_mask_avg_3c_int)
        MC_mask_union_dice_with_SI = wjb_functions.dice(MC_mask_union_3c_int,SI_mask_2c)
        
        PERT_mask_avg_dice_with_SI = wjb_functions.dice(PERT_mask_avg_3c_int,SI_mask_2c)
        PERT_mask_union_dice_with_avg = wjb_functions.dice(PERT_mask_union_3c_int,PERT_mask_avg_3c_int)
        PERT_mask_union_dice_with_SI = wjb_functions.dice(PERT_mask_union_3c_int,SI_mask_2c)



        MC_mask_th_dice_3c_array_with_SI = []
        MC_mask_th_dice_3c_array_with_avg = []
        PERT_mask_th_dice_3c_array_with_SI = []
        PERT_mask_th_dice_3c_array_with_avg = []
        th_range = np.arange(0,11,1)/10
        
        y_with_SI=[]
        y_with_avg=[]
        for th in th_range:
            tau_array_3c = np.copy(MC_tau_array_3c)
            obj_3c = np.where(tau_array_3c<=th)[0]
            if obj_3c.size == 0: 
                mask_th_3c = np.zeros([MC_label_image.shape[0],MC_label_image.shape[1]])
                mask_th_3c_no_dilation = np.zeros([MC_label_image.shape[0],MC_label_image.shape[1]])
            else:
                mask_th_3c_temp = MC_mask_matrix[:,:,obj_3c]
                mask_th_3c_no_dilation = np.sum(mask_th_3c_temp,axis=-1)
                mask_th_3c = wjb_functions.dilation(mask_th_3c_no_dilation, radius)
            
            MC_mask_th_dice_3c_with_SI = wjb_functions.dice(SI_mask_2c, mask_th_3c)
            MC_mask_th_dice_3c_with_avg = wjb_functions.dice(MC_mask_union_3c_int, mask_th_3c)
            MC_mask_th_dice_3c_array_with_SI.append(MC_mask_th_dice_3c_with_SI)
            MC_mask_th_dice_3c_array_with_avg.append(MC_mask_th_dice_3c_with_avg)

            del mask_th_3c
            del tau_array_3c
            
            tau_array_3c = np.copy(PERT_tau_array_3c)
            obj_3c = np.where(tau_array_3c<=th)[0]
            if obj_3c.size == 0: 
                mask_th_3c = np.zeros([PERT_label_image.shape[0],PERT_label_image.shape[1]])
                mask_th_3c_no_dilation = np.zeros([PERT_label_image.shape[0],PERT_label_image.shape[1]])
            else:
                mask_th_3c_temp = PERT_mask_matrix[:,:,obj_3c]
                mask_th_3c_no_dilation = np.sum(mask_th_3c_temp,axis=-1)
                mask_th_3c = wjb_functions.dilation(mask_th_3c_no_dilation, radius)
            
            PERT_mask_th_dice_3c_with_SI = wjb_functions.dice(SI_mask_2c, mask_th_3c)
            PERT_mask_th_dice_3c_with_avg = wjb_functions.dice(PERT_mask_union_3c_int, mask_th_3c)
            PERT_mask_th_dice_3c_array_with_SI.append(PERT_mask_th_dice_3c_with_SI)
            PERT_mask_th_dice_3c_array_with_avg.append(PERT_mask_th_dice_3c_with_avg)
            
            dice_array_image_with_SI = [MC_mask_th_dice_3c_with_SI, PERT_mask_th_dice_3c_with_SI]
            dice_array_image_with_avg = [MC_mask_th_dice_3c_with_avg, PERT_mask_th_dice_3c_with_avg]
            
            max_dice_with_SI = np.max(dice_array_image_with_SI)
            max_dice_with_avg = np.max(dice_array_image_with_avg)
            y_with_SI.append(max_dice_with_SI)
            y_with_avg.append(max_dice_with_avg)
            
        SI_dice_array.append(SI_dice_2c)
        rho_tra_le_curve_MC.append(np.corrcoef(MC_mask_th_dice_3c_array_with_SI,MC_mask_th_dice_3c_array_with_avg)[0][1])
        rho_tra_le_curve_PERT.append(np.corrcoef(PERT_mask_th_dice_3c_array_with_SI,PERT_mask_th_dice_3c_array_with_avg)[0][1])
        rho_tra_le_curve_y.append(np.corrcoef(y_with_SI,y_with_avg)[0][1])
        
        eucl_dist_MC.append(np.linalg.norm([MC_mask_th_dice_3c_array_with_SI,MC_mask_th_dice_3c_array_with_avg]))
        eucl_dist_PERT.append(np.linalg.norm([PERT_mask_th_dice_3c_array_with_SI,PERT_mask_th_dice_3c_array_with_avg]))
        eucl_dist_y.append(np.linalg.norm([y_with_SI,y_with_avg]))
        
        monotony_corr_avg_y.append(np.corrcoef(th_range,y_with_avg)[0][1])
        monotony_corr_SI_y.append(np.corrcoef(th_range,y_with_SI)[0][1])
        
        monotony_corr_avg_MC.append(np.corrcoef(th_range,MC_mask_th_dice_3c_array_with_avg)[0][1])
        monotony_corr_SI_MC.append(np.corrcoef(th_range,MC_mask_th_dice_3c_array_with_SI)[0][1])
        
        monotony_corr_avg_PERT.append(np.corrcoef(th_range,PERT_mask_th_dice_3c_array_with_avg)[0][1])
        monotony_corr_SI_PERT.append(np.corrcoef(th_range,PERT_mask_th_dice_3c_array_with_SI)[0][1])

        if not os.path.isdir(general_path_to_save + subset + "/"): os.makedirs(general_path_to_save + subset + "/")
        plt.figure(figsize=(28,8))
        plt.subplot(131)
        plt.title("Montecarlo")
        plt.plot(th_range,MC_mask_th_dice_3c_array_with_SI,'b',label='Dice per threshold, with GT = SI')
        plt.plot(th_range,MC_mask_th_dice_3c_array_with_avg,'m', label='Dice per threshold, with GT = mask avg')
        plt.axhline(MC_mask_union_dice_with_SI, color='b',label='Baseline UNION vs SI', linestyle='--')
        plt.axhline(MC_mask_union_dice_with_avg, color='m',label='Baseline UNION vs avg', linestyle='--')
        plt.axhline(SI_dice_2c, color='r',label='Baseline SI vs GT', linestyle='--')
        plt.axhline(MC_mask_avg_dice_with_SI, color='g',label='Baseline SI vs avg', linestyle='--')
        plt.legend(loc=4)
        plt.xlim(0,1)
        plt.ylim(0)
        plt.subplot(132)
        plt.title("Perturbation")
        plt.plot(th_range,PERT_mask_th_dice_3c_array_with_SI,'b',label='Dice per threshold, with GT = SI')
        plt.plot(th_range,PERT_mask_th_dice_3c_array_with_avg,'m', label='Dice per threshold, with GT = mask avg')
        plt.axhline(PERT_mask_union_dice_with_SI, color='b',label='Baseline UNION vs SI', linestyle='--')
        plt.axhline(PERT_mask_union_dice_with_avg, color='m',label='Baseline UNION vs avg', linestyle='--')
        plt.axhline(SI_dice_2c, color='r',label='Baseline SI vs GT', linestyle='--')
        plt.axhline(PERT_mask_avg_dice_with_SI, color='g',label='Baseline SI vs avg', linestyle='--')
        plt.legend(loc=4)
        plt.xlim(0,1)
        plt.ylim(0)
        plt.subplot(133)
        plt.title("Max of MC or Pert")
        plt.plot(th_range,y_with_SI,'b',label='Dice per threshold, with GT = SI')
        plt.plot(th_range,y_with_avg,'m', label='Dice per threshold, with GT = mask avg')
        plt.axhline(SI_dice_2c, color='r',label='Baseline SI vs GT', linestyle='--')
        plt.legend(loc=4)
        plt.xlim(0,1)
        plt.ylim(0)
        plt.savefig(general_path_to_save + subset + "/" + image)
        plt.close()
    
    corr_MC_rho = np.corrcoef(SI_dice_array,rho_tra_le_curve_MC)[0][1]
    corr_PERT_rho = np.corrcoef(SI_dice_array,rho_tra_le_curve_PERT)[0][1]
    corr_y_rho = np.corrcoef(SI_dice_array,rho_tra_le_curve_y)[0][1]
    
    corr_MC_eucl = np.corrcoef(SI_dice_array,eucl_dist_MC)[0][1]
    corr_PERT_eucl = np.corrcoef(SI_dice_array,eucl_dist_PERT)[0][1]
    corr_y_eucl = np.corrcoef(SI_dice_array,eucl_dist_y)[0][1]
    
    corr_MC_monotony_SI = np.corrcoef(SI_dice_array,monotony_corr_SI_MC)[0][1]
    corr_PERT_monotony_SI = np.corrcoef(SI_dice_array,monotony_corr_SI_PERT)[0][1]
    corr_y_monotony_SI = np.corrcoef(SI_dice_array,monotony_corr_SI_y)[0][1]
    corr_MC_monotony_avg = np.corrcoef(SI_dice_array,monotony_corr_avg_MC)[0][1]
    corr_PERT_monotony_avg = np.corrcoef(SI_dice_array,monotony_corr_avg_PERT)[0][1]
    corr_y_monotony_avg = np.corrcoef(SI_dice_array,monotony_corr_avg_y)[0][1]
    
    SI_dice_array = np.array(SI_dice_array)
    rho_tra_le_curve_MC = np.array(rho_tra_le_curve_MC)
    rho_tra_le_curve_PERT = np.array(rho_tra_le_curve_PERT)
    rho_tra_le_curve_y = np.array(rho_tra_le_curve_y)
    
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.title("Correlation between rho of the curves and dice")
    plt.scatter(SI_dice_array,rho_tra_le_curve_MC, color='b', label="MC, rho=" + str(corr_MC_rho)[:4])
    plt.scatter(SI_dice_array,rho_tra_le_curve_PERT, color='g', label="PERT, rho=" + str(corr_PERT_rho)[:4])
    plt.scatter(SI_dice_array,rho_tra_le_curve_y, color='m', label="Max tra MC e PERT, rho=" + str(corr_y_rho)[:4])
    plt.xlim(0,1)
    plt.legend()
    plt.subplot(122)
    plt.title("Correlation between eucl distance of the curves and dice")
    plt.scatter(SI_dice_array,eucl_dist_MC, color='b', label="MC, rho=" + str(corr_MC_eucl)[:4])
    plt.scatter(SI_dice_array,eucl_dist_PERT, color='g', label="PERT, rho=" + str(corr_PERT_eucl)[:4])
    plt.scatter(SI_dice_array,eucl_dist_y, color='m', label="Max tra MC e PERT, rho=" + str(corr_y_eucl)[:4])
    plt.xlim(0,1)
    plt.legend()
    plt.savefig(general_path_to_save + subset + "/correlations.png")
    plt.close()
    
    corr_mon_SI_SI_dice_MC = np.corrcoef(SI_dice_array,monotony_corr_SI_MC)[0][1]
    corr_mon_SI_SI_dice_PERT = np.corrcoef(SI_dice_array,monotony_corr_SI_PERT)[0][1]
    corr_mon_SI_SI_dice_y = np.corrcoef(SI_dice_array,monotony_corr_SI_y)[0][1]
    corr_mon_avg_SI_dice_MC = np.corrcoef(SI_dice_array,monotony_corr_avg_MC)[0][1]
    corr_mon_avg_SI_dice_PERT = np.corrcoef(SI_dice_array,monotony_corr_avg_PERT)[0][1]
    corr_mon_avg_SI_dice_y = np.corrcoef(SI_dice_array,monotony_corr_avg_y)[0][1]
    
    
    
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.title("Correlation between monotony of the curves and dice (SI)")
    plt.scatter(SI_dice_array,monotony_corr_SI_MC, color='b', label="MC, rho=" + str(corr_mon_SI_SI_dice_MC)[:4])
    plt.scatter(SI_dice_array,monotony_corr_SI_PERT, color='g', label="PERT, rho=" + str(corr_mon_SI_SI_dice_PERT)[:4])
    plt.scatter(SI_dice_array,monotony_corr_SI_y, color='m', label="Max tra MC e PERT, rho=" + str(corr_mon_SI_SI_dice_y)[:4])
    plt.xlim(0,1)
    plt.legend()
    plt.subplot(122)
    plt.title("Correlation between monotony of the curves and dice (avg)")
    plt.scatter(SI_dice_array,monotony_corr_avg_MC, color='b', label="MC, rho=" + str(corr_mon_avg_SI_dice_MC)[:4])
    plt.scatter(SI_dice_array,monotony_corr_avg_PERT, color='g', label="PERT, rho=" + str(corr_mon_avg_SI_dice_PERT)[:4])
    plt.scatter(SI_dice_array,monotony_corr_avg_y, color='m', label="Max tra MC e PERT, rho=" + str(corr_mon_avg_SI_dice_y)[:4])
    plt.xlim(0,1)
    plt.legend()
    plt.savefig(general_path_to_save + subset + "/correlations_monotony.png")
    plt.close()