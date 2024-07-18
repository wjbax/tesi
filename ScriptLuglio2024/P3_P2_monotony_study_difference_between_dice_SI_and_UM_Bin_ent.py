# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:00:11 2024

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
# diff_type_name = "_perturbation/"
# dataset = "Liver HE steatosis/"
# subset = "test"
# image = "1004289_35.png"
N = 20
# radius = 5
c = 3

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
# for diff_type in tqdm(["PERT","MC"]):
#     if diff_type == "PERT": diff_type_name = "_perturbation"
#     if diff_type == "MC": diff_type_name = "_MC"

for dataset in ["Renal PAS glomeruli/", "Renal PAS tubuli/"]:
    
    general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V23/"
    general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
    general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    
    for subset in tqdm(["test", "val"]):
                
        GT_dict_max_values_MC_C1 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        GT_dict_max_values_PERT_C1 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        GT_dict_max_values_MC_C2 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        GT_dict_max_values_PERT_C2 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        # SI_dict_max_values_MC_C1 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # SI_dict_max_values_PERT_C1 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # SI_dict_max_values_MC_C2 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # SI_dict_max_values_PERT_C2 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_PERT_dict_max_values_MC_C1 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_PERT_dict_max_values_PERT_C1 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_PERT_dict_max_values_MC_C2 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_PERT_dict_max_values_PERT_C2 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_MC_dict_max_values_MC_C1 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_MC_dict_max_values_PERT_C1 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_MC_dict_max_values_MC_C2 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # avg_MC_dict_max_values_PERT_C2 = {
        #     'image_name': [],
        #     'union_max_dice': [],
        #     'th_union_max_dice': [],
        #     'flag_union_dice': []
        #     }
        
        # rho_GT_SI_MC_C1_array = []
        # rho_GT_SI_PERT_C1_array = []
        # rho_GT_avg_PERT_MC_C1_array = []
        # rho_GT_avg_PERT_PERT_C1_array = []
        # rho_GT_avg_MC_MC_C1_array = []
        # rho_GT_avg_MC_PERT_C1_array = []
        
        # rho_GT_SI_MC_C2_array = []
        # rho_GT_SI_PERT_C2_array = []
        # rho_GT_avg_PERT_MC_C2_array = []
        # rho_GT_avg_PERT_PERT_C2_array = []
        # rho_GT_avg_MC_MC_C2_array = []
        # rho_GT_avg_MC_PERT_C2_array = []
        
        # rho_GT_SI_max_MC_PERT_C1_array = []
        # rho_GT_SI_max_MC_PERT_C2_array = []
        # rho_GT_avg_PERT_max_MC_PERT_C1_array = []
        # rho_GT_avg_PERT_max_MC_PERT_C2_array = []
        # rho_GT_avg_MC_max_MC_PERT_C1_array = []
        # rho_GT_avg_MC_max_MC_PERT_C2_array = []
        
        GT_monotony_MC_C1_array = []
        GT_monotony_MC_C2_array = []
        GT_monotony_PERT_C1_array = []
        GT_monotony_PERT_C2_array = []
        
        # SI_monotony_MC_C1_array = []
        # SI_monotony_MC_C2_array = []
        # SI_monotony_PERT_C1_array = []
        # SI_monotony_PERT_C2_array = []
        
        # avg_PERT_monotony_MC_C1_array = []
        # avg_PERT_monotony_MC_C2_array = []
        # avg_PERT_monotony_PERT_C1_array = []
        # avg_PERT_monotony_PERT_C2_array = []
        
        # avg_MC_monotony_MC_C1_array = []
        # avg_MC_monotony_MC_C2_array = []
        # avg_MC_monotony_PERT_C1_array = []
        # avg_MC_monotony_PERT_C2_array = []
        
        GT_monotony_max_MC_PERT_C1_array = []
        GT_monotony_max_MC_PERT_C2_array = []
        
        # SI_monotony_max_MC_PERT_C1_array = []
        # SI_monotony_max_MC_PERT_C2_array = []
        
        # avg_PERT_monotony_max_MC_PERT_C1_array = []
        # avg_PERT_monotony_max_MC_PERT_C2_array = []
        
        # avg_MC_monotony_max_MC_PERT_C1_array = []
        # avg_MC_monotony_max_MC_PERT_C2_array = []
        
        GT_SI_dice_array_C1 = []
        GT_SI_dice_array_C2 = []
        
        
        SI_UM_dice_MC_C1_array = []
        SI_UM_dice_MC_C2_array = []
        SI_UM_dice_PERT_C1_array = []
        SI_UM_dice_PERT_C2_array = []
        
        SI_UM_max_dice_MC_C1_array = []
        SI_UM_max_dice_MC_C2_array = []
        SI_UM_max_dice_PERT_C1_array = []
        SI_UM_max_dice_PERT_C2_array = []
        
        for image in tqdm(os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/")):
            
            # image = '1004761_25.png'
            
            GT_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
            SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
            diff_path_3c_MC = general_results_path_3c + "_MC" + "/" + subset + "/" + "mask/" + image + "/"
            softmax_path_3c_MC = general_results_path_3c + "_MC" + "/" + subset + "/" + "softmax/" + image + "/"
            diff_path_3c_PERT = general_results_path_3c + "_perturbation" + "/" + subset + "/" + "mask/" + image + "/"
            softmax_path_3c_PERT = general_results_path_3c + "_perturbation" + "/" + subset + "/" + "softmax/" + image + "/"
            
            
            #%%
            dataset_name = dataset[:-1] + "_3c/"
            GT_mask_original = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
            GT_mask_C1,GT_mask_C2 = wjb_functions.mask_splitter(GT_mask_original)
            
            C1_exist = True
            C2_exist = True
            if not GT_mask_C1.any(): C1_exist = False
            if not GT_mask_C2.any(): C2_exist = False
            
            if not C1_exist and not C2_exist: 
                print("Image " + image[:-4] + " has no GT")
                continue
            
            mask_union_MC = wjb_functions.mask_union_gen_3c(diff_path_3c_MC)/255
            mask_union_C1_MC,mask_union_C2_MC = wjb_functions.mask_splitter(mask_union_MC)
            mask_union_PERT = wjb_functions.mask_union_gen_3c(diff_path_3c_PERT)/255
            mask_union_C1_PERT,mask_union_C2_PERT = wjb_functions.mask_splitter(mask_union_PERT)
            SI_mask_original = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
            SI_mask_C1,SI_mask_C2 = wjb_functions.mask_splitter(SI_mask_original)
            
            SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
            SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
            
            DIM = np.shape(GT_mask_original)[:2]
            softmax_matrix_MC = wjb_functions.softmax_matrix_gen(softmax_path_3c_MC, DIM, c, N)
            unc_map_C1_MC = wjb_functions.binary_entropy_map(softmax_matrix_MC[:,:,1,:])
            unc_map_C2_MC = wjb_functions.binary_entropy_map(softmax_matrix_MC[:,:,2,:])
            softmax_matrix_PERT = wjb_functions.softmax_matrix_gen(softmax_path_3c_PERT, DIM, c, N)
            unc_map_C1_PERT = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
            unc_map_C2_PERT = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,2,:])
            
            # unc_map_C1_PERT = wjb_functions.mean_softmax_BC_3(softmax_matrix_PERT)
            # unc_map_C1_MC = wjb_functions.mean_softmax_BC_3(softmax_matrix_MC)
            # unc_map_C2_PERT = copy.deepcopy(unc_map_C1_PERT)
            # unc_map_C2_MC = copy.deepcopy(unc_map_C1_MC)
            
            mask_avg_PERT = wjb_functions.mask_avg_gen_3c(softmax_matrix_PERT)/2
            avg_PERT_mask_C1,avg_PERT_mask_C2 = wjb_functions.mask_splitter(mask_avg_PERT)
            mask_avg_MC = wjb_functions.mask_avg_gen_3c(softmax_matrix_MC)/2
            avg_MC_mask_C1,avg_MC_mask_C2 = wjb_functions.mask_splitter(mask_avg_MC)

            th_range = np.arange(1,11,1)/10
            
            GT_union_th_dice_array_C1_MC = []
            GT_union_th_dice_array_C2_MC = []
            GT_union_th_dice_array_C1_PERT = []
            GT_union_th_dice_array_C2_PERT = []
            
            # SI_union_th_dice_array_C1_MC = []
            # SI_union_th_dice_array_C2_MC = []
            # SI_union_th_dice_array_C1_PERT = []
            # SI_union_th_dice_array_C2_PERT = []
            
            # avg_PERT_union_th_dice_array_C1_MC = []
            # avg_PERT_union_th_dice_array_C2_MC = []
            # avg_PERT_union_th_dice_array_C1_PERT = []
            # avg_PERT_union_th_dice_array_C2_PERT = []
            
            # avg_MC_union_th_dice_array_C1_MC = []
            # avg_MC_union_th_dice_array_C2_MC = []
            # avg_MC_union_th_dice_array_C1_PERT = []
            # avg_MC_union_th_dice_array_C2_PERT = []
            
            
            
            # C1           
            
            UM_max_dice_MC_C1 = np.zeros_like(GT_mask_C1)
            UM_max_dice_MC_C2 = np.zeros_like(GT_mask_C1)
            UM_max_dice_PERT_C1 = np.zeros_like(GT_mask_C1)
            UM_max_dice_PERT_C2 = np.zeros_like(GT_mask_C1)
            
            max_dice_MC_C1 = 0
            max_dice_MC_C2 = 0
            max_dice_PERT_C1 = 0
            max_dice_PERT_C2 = 0
            
            for th in th_range:
                GT_mask_th_union_C1_MC = mask_union_C1_MC.astype(bool) & (unc_map_C1_MC<=th)
                dice = wjb_functions.dice(GT_mask_th_union_C1_MC, GT_mask_C1)
                GT_union_th_dice_array_C1_MC.append(dice)
                if dice > max_dice_MC_C1: 
                    max_dice_MC_C1 = dice
                    UM_max_dice_MC_C1 = copy.deepcopy(GT_mask_th_union_C1_MC)
                del dice
                
                GT_mask_th_union_C2_MC = mask_union_C2_MC.astype(bool) & (unc_map_C2_MC<=th)
                dice = wjb_functions.dice(GT_mask_th_union_C2_MC, GT_mask_C2)
                GT_union_th_dice_array_C2_MC.append(dice)
                if dice > max_dice_MC_C2: 
                    max_dice_MC_C2 = dice
                    UM_max_dice_MC_C2 = copy.deepcopy(GT_mask_th_union_C2_MC)
                del dice
                
                GT_mask_th_union_C1_PERT = mask_union_C1_PERT.astype(bool) & (unc_map_C1_PERT<=th)
                dice = wjb_functions.dice(GT_mask_th_union_C1_PERT, GT_mask_C1)
                GT_union_th_dice_array_C1_PERT.append(dice)
                if dice > max_dice_PERT_C1: 
                    max_dice_PERT_C1 = dice
                    UM_max_dice_PERT_C1 = copy.deepcopy(GT_mask_th_union_C1_PERT)
                del dice
                
                GT_mask_th_union_C2_PERT = mask_union_C2_PERT.astype(bool) & (unc_map_C2_PERT<=th)
                dice = wjb_functions.dice(GT_mask_th_union_C2_PERT, GT_mask_C2)
                GT_union_th_dice_array_C2_PERT.append(dice)
                if dice > max_dice_PERT_C2: 
                    max_dice_PERT_C2 = dice
                    UM_max_dice_PERT_C2 = copy.deepcopy(GT_mask_th_union_C2_PERT)
                del dice
                
                
                
                # SI_mask_th_union_C1_MC = mask_union_C1_MC.astype(bool) & (unc_map_C1_MC<=th)
                # SI_union_th_dice_array_C1_MC.append(wjb_functions.dice(SI_mask_th_union_C1_MC, SI_mask_C1))
                # SI_mask_th_union_C2_MC = mask_union_C2_MC.astype(bool) & (unc_map_C2_MC<=th)
                # SI_union_th_dice_array_C2_MC.append(wjb_functions.dice(SI_mask_th_union_C2_MC, SI_mask_C2))
                # SI_mask_th_union_C1_PERT = mask_union_C1_PERT.astype(bool) & (unc_map_C1_PERT<=th)
                # SI_union_th_dice_array_C1_PERT.append(wjb_functions.dice(SI_mask_th_union_C1_PERT, SI_mask_C1))
                # SI_mask_th_union_C2_PERT = mask_union_C2_PERT.astype(bool) & (unc_map_C2_PERT<=th)
                # SI_union_th_dice_array_C2_PERT.append(wjb_functions.dice(SI_mask_th_union_C2_PERT, SI_mask_C2))
                
                # avg_PERT_mask_th_union_C1_MC = mask_union_C1_MC.astype(bool) & (unc_map_C1_MC<=th)
                # avg_PERT_union_th_dice_array_C1_MC.append(wjb_functions.dice(avg_PERT_mask_th_union_C1_MC, avg_PERT_mask_C1))
                # avg_PERT_mask_th_union_C2_MC = mask_union_C2_MC.astype(bool) & (unc_map_C2_MC<=th)
                # avg_PERT_union_th_dice_array_C2_MC.append(wjb_functions.dice(avg_PERT_mask_th_union_C2_MC, avg_PERT_mask_C2))
                # avg_PERT_mask_th_union_C1_PERT = mask_union_C1_PERT.astype(bool) & (unc_map_C1_PERT<=th)
                # avg_PERT_union_th_dice_array_C1_PERT.append(wjb_functions.dice(avg_PERT_mask_th_union_C1_PERT, avg_PERT_mask_C1))
                # avg_PERT_mask_th_union_C2_PERT = mask_union_C2_PERT.astype(bool) & (unc_map_C2_PERT<=th)
                # avg_PERT_union_th_dice_array_C2_PERT.append(wjb_functions.dice(avg_PERT_mask_th_union_C2_PERT, avg_PERT_mask_C2))
                
                # avg_MC_mask_th_union_C1_MC = mask_union_C1_MC.astype(bool) & (unc_map_C1_MC<=th)
                # avg_MC_union_th_dice_array_C1_MC.append(wjb_functions.dice(avg_MC_mask_th_union_C1_MC, avg_MC_mask_C1))
                # avg_MC_mask_th_union_C2_MC = mask_union_C2_MC.astype(bool) & (unc_map_C2_MC<=th)
                # avg_MC_union_th_dice_array_C2_MC.append(wjb_functions.dice(avg_MC_mask_th_union_C2_MC, avg_MC_mask_C2))
                # avg_MC_mask_th_union_C1_PERT = mask_union_C1_PERT.astype(bool) & (unc_map_C1_PERT<=th)
                # avg_MC_union_th_dice_array_C1_PERT.append(wjb_functions.dice(avg_MC_mask_th_union_C1_PERT, avg_MC_mask_C1))
                # avg_MC_mask_th_union_C2_PERT = mask_union_C2_PERT.astype(bool) & (unc_map_C2_PERT<=th)
                # avg_MC_union_th_dice_array_C2_PERT.append(wjb_functions.dice(avg_MC_mask_th_union_C2_PERT, avg_MC_mask_C2))
                
            GT_union_th_dice_array_C1_MC = np.array(GT_union_th_dice_array_C1_MC)
            GT_union_th_dice_array_C2_MC = np.array(GT_union_th_dice_array_C2_MC)
            GT_union_th_dice_array_C1_PERT = np.array(GT_union_th_dice_array_C1_PERT)
            GT_union_th_dice_array_C2_PERT = np.array(GT_union_th_dice_array_C2_PERT)
            
            # SI_union_th_dice_array_C1_MC = np.array(SI_union_th_dice_array_C1_MC)
            # SI_union_th_dice_array_C2_MC = np.array(SI_union_th_dice_array_C2_MC)
            # SI_union_th_dice_array_C1_PERT = np.array(SI_union_th_dice_array_C1_PERT)
            # SI_union_th_dice_array_C2_PERT = np.array(SI_union_th_dice_array_C2_PERT)
            
            # avg_PERT_union_th_dice_array_C1_MC = np.array(avg_PERT_union_th_dice_array_C1_MC)
            # avg_PERT_union_th_dice_array_C2_MC = np.array(avg_PERT_union_th_dice_array_C2_MC)
            # avg_PERT_union_th_dice_array_C1_PERT = np.array(avg_PERT_union_th_dice_array_C1_PERT)
            # avg_PERT_union_th_dice_array_C2_PERT = np.array(avg_PERT_union_th_dice_array_C2_PERT)
            
            # avg_MC_union_th_dice_array_C1_MC = np.array(avg_MC_union_th_dice_array_C1_MC)
            # avg_MC_union_th_dice_array_C2_MC = np.array(avg_MC_union_th_dice_array_C2_MC)
            # avg_MC_union_th_dice_array_C1_PERT = np.array(avg_MC_union_th_dice_array_C1_PERT)
            # avg_MC_union_th_dice_array_C2_PERT = np.array(avg_MC_union_th_dice_array_C2_PERT)
            
            path_to_save_figure = general_path_to_save + dataset_name + subset + "/"
            if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
            
            tabella_temp = np.array([GT_union_th_dice_array_C1_MC,
                                     GT_union_th_dice_array_C2_MC,
                                     GT_union_th_dice_array_C1_PERT,
                                     GT_union_th_dice_array_C2_PERT]).T
            
            df_temp = pd.DataFrame({'image': image,
                                    'union_th_dice_array_C1_MC': tabella_temp[:, 0],
                                    'union_th_dice_array_C2_MC': tabella_temp[:, 1],
                                    'union_th_dice_array_C1_PERT': tabella_temp[:, 2],
                                    'union_th_dice_array_C2_PERT': tabella_temp[:, 3]
                                    })
            
            if C1_exist:
                
                GT_dict_max_values_MC_C1['image_name'].append(image)
                GT_dict_max_values_MC_C1['union_max_dice'].append(np.nanmax(GT_union_th_dice_array_C1_MC))
                GT_dict_max_values_MC_C1['th_union_max_dice'].append(th_range[np.where(GT_union_th_dice_array_C1_MC == np.nanmax(GT_union_th_dice_array_C1_MC))[0][0]])
                
                if np.max(GT_union_th_dice_array_C1_MC) > SI_dice_C1: 
                    GT_dict_max_values_MC_C1['flag_union_dice'].append(1)
                else: 
                    GT_dict_max_values_MC_C1['flag_union_dice'].append(0)
                    
                df_gen = pd.DataFrame.from_dict(GT_dict_max_values_MC_C1)
                df_gen.to_csv(path_to_save_figure + "GT_DATAFRAME_max_values_MC_C1.csv", index=True)
                del df_gen
                
                GT_dict_max_values_PERT_C1['image_name'].append(image)
                GT_dict_max_values_PERT_C1['union_max_dice'].append(np.nanmax(GT_union_th_dice_array_C1_PERT))
                GT_dict_max_values_PERT_C1['th_union_max_dice'].append(th_range[np.where(GT_union_th_dice_array_C1_PERT == np.nanmax(GT_union_th_dice_array_C1_PERT))[0][0]])
                
                if np.max(GT_union_th_dice_array_C1_PERT) > SI_dice_C1: 
                    GT_dict_max_values_PERT_C1['flag_union_dice'].append(1)
                else: 
                    GT_dict_max_values_PERT_C1['flag_union_dice'].append(0)
                    
                df_gen = pd.DataFrame.from_dict(GT_dict_max_values_PERT_C1)
                df_gen.to_csv(path_to_save_figure + "GT_DATAFRAME_max_values_PERT_C1.csv", index=True)
                del df_gen
                
                title_C1 = "Classe 1"
            else: title_C1 = "Classe 1 - Not present in GT"
                
            if C2_exist:
                GT_dict_max_values_MC_C2['image_name'].append(image)
                GT_dict_max_values_MC_C2['union_max_dice'].append(np.nanmax(GT_union_th_dice_array_C2_MC))
                GT_dict_max_values_MC_C2['th_union_max_dice'].append(th_range[np.where(GT_union_th_dice_array_C2_MC == np.nanmax(GT_union_th_dice_array_C2_MC))[0][0]])
                
                if np.max(GT_union_th_dice_array_C2_MC) > SI_dice_C2: 
                    GT_dict_max_values_MC_C2['flag_union_dice'].append(2)
                else: 
                    GT_dict_max_values_MC_C2['flag_union_dice'].append(0)
                    
                df_gen = pd.DataFrame.from_dict(GT_dict_max_values_MC_C2)
                df_gen.to_csv(path_to_save_figure + "GT_DATAFRAME_max_values_MC_C2.csv", index=True)
                del df_gen
                    
                GT_dict_max_values_PERT_C2['image_name'].append(image)
                GT_dict_max_values_PERT_C2['union_max_dice'].append(np.nanmax(GT_union_th_dice_array_C2_PERT))
                GT_dict_max_values_PERT_C2['th_union_max_dice'].append(th_range[np.where(GT_union_th_dice_array_C2_PERT == np.nanmax(GT_union_th_dice_array_C2_PERT))[0][0]])
                
                if np.max(GT_union_th_dice_array_C2_PERT) > SI_dice_C2: 
                    GT_dict_max_values_PERT_C2['flag_union_dice'].append(2)
                else: 
                    GT_dict_max_values_PERT_C2['flag_union_dice'].append(0)
                
                df_gen = pd.DataFrame.from_dict(GT_dict_max_values_PERT_C2)
                df_gen.to_csv(path_to_save_figure + "GT_DATAFRAME_max_values_PERT_C2.csv", index=True)
                del df_gen
                title_C2 = "Classe 2"
            else: title_C2 = "Classe 2 - Not present in GT"
            
            
            # tabella_temp = np.array([SI_union_th_dice_array_C1_MC,
            #                          SI_union_th_dice_array_C2_MC,
            #                          SI_union_th_dice_array_C1_PERT,
            #                          SI_union_th_dice_array_C2_PERT]).T
            
            # df_temp = pd.DataFrame({'image': image,
            #                         'union_th_dice_array_C1_MC': tabella_temp[:, 0],
            #                         'union_th_dice_array_C2_MC': tabella_temp[:, 1],
            #                         'union_th_dice_array_C1_PERT': tabella_temp[:, 2],
            #                         'union_th_dice_array_C2_PERT': tabella_temp[:, 3]
            #                         })
            
            # if C1_exist:
            #     SI_dict_max_values_MC_C1['image_name'].append(image)
            #     SI_dict_max_values_MC_C1['union_max_dice'].append(np.nanmax(SI_union_th_dice_array_C1_MC))
            #     if not np.nanmax(SI_union_th_dice_array_C1_MC) >= 0: SI_dict_max_values_MC_C1['th_union_max_dice'].append(np.nan)
            #     else: SI_dict_max_values_MC_C1['th_union_max_dice'].append(th_range[np.where(SI_union_th_dice_array_C1_MC == np.nanmax(SI_union_th_dice_array_C1_MC))[0][0]])
                
            #     if np.max(SI_union_th_dice_array_C1_MC) > SI_dice_C1: 
            #         SI_dict_max_values_MC_C1['flag_union_dice'].append(1)
            #     else: 
            #         SI_dict_max_values_MC_C1['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(SI_dict_max_values_MC_C1)
            #     df_gen.to_csv(path_to_save_figure + "SI_DATAFRAME_max_values_MC_C1.csv", index=True)
            #     del df_gen
                
            #     SI_dict_max_values_PERT_C1['image_name'].append(image)
            #     SI_dict_max_values_PERT_C1['union_max_dice'].append(np.nanmax(SI_union_th_dice_array_C1_PERT))
            #     if not np.nanmax(SI_union_th_dice_array_C1_PERT) >= 0: SI_dict_max_values_PERT_C1['th_union_max_dice'].append(np.nan)
            #     else: SI_dict_max_values_PERT_C1['th_union_max_dice'].append(th_range[np.where(SI_union_th_dice_array_C1_PERT == np.nanmax(SI_union_th_dice_array_C1_PERT))[0][0]])
                
            #     if np.max(SI_union_th_dice_array_C1_PERT) > SI_dice_C1: 
            #         SI_dict_max_values_PERT_C1['flag_union_dice'].append(1)
            #     else: 
            #         SI_dict_max_values_PERT_C1['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(SI_dict_max_values_PERT_C1)
            #     df_gen.to_csv(path_to_save_figure + "SI_DATAFRAME_max_values_PERT_C1.csv", index=True)
            #     del df_gen
                
            #     title_C1 = "Classe 1"
            # else: title_C1 = "Classe 1 - Not present in SI"
                
            # if C2_exist:
                
            #     SI_dict_max_values_MC_C2['image_name'].append(image)
            #     SI_dict_max_values_MC_C2['union_max_dice'].append(np.nanmax(SI_union_th_dice_array_C2_MC))
            #     if not np.nanmax(SI_union_th_dice_array_C2_MC) >= 0: SI_dict_max_values_MC_C2['th_union_max_dice'].append(np.nan)
            #     else: SI_dict_max_values_MC_C2['th_union_max_dice'].append(th_range[np.where(SI_union_th_dice_array_C2_MC == np.nanmax(SI_union_th_dice_array_C2_MC))[0][0]])
                
            #     if np.max(SI_union_th_dice_array_C2_MC) > SI_dice_C2: 
            #         SI_dict_max_values_MC_C2['flag_union_dice'].append(2)
            #     else: 
            #         SI_dict_max_values_MC_C2['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(SI_dict_max_values_MC_C2)
            #     df_gen.to_csv(path_to_save_figure + "SI_DATAFRAME_max_values_MC_C2.csv", index=True)
            #     del df_gen
                    
            #     SI_dict_max_values_PERT_C2['image_name'].append(image)
            #     SI_dict_max_values_PERT_C2['union_max_dice'].append(np.nanmax(SI_union_th_dice_array_C2_PERT))
            #     if not np.nanmax(SI_union_th_dice_array_C2_PERT) >= 0: SI_dict_max_values_PERT_C2['th_union_max_dice'].append(np.nan)
            #     else: SI_dict_max_values_PERT_C2['th_union_max_dice'].append(th_range[np.where(SI_union_th_dice_array_C2_PERT == np.nanmax(SI_union_th_dice_array_C2_PERT))[0][0]])
                
            #     if np.max(SI_union_th_dice_array_C2_PERT) > SI_dice_C2: 
            #         SI_dict_max_values_PERT_C2['flag_union_dice'].append(2)
            #     else: 
            #         SI_dict_max_values_PERT_C2['flag_union_dice'].append(0)
                
            #     df_gen = pd.DataFrame.from_dict(SI_dict_max_values_PERT_C2)
            #     df_gen.to_csv(path_to_save_figure + "SI_DATAFRAME_max_values_PERT_C2.csv", index=True)
            #     del df_gen
            #     title_C2 = "Classe 2"
            # else: title_C2 = "Classe 2 - Not present in SI"
            
            
            # tabella_temp = np.array([avg_PERT_union_th_dice_array_C1_MC,
            #                          avg_PERT_union_th_dice_array_C2_MC,
            #                          avg_PERT_union_th_dice_array_C1_PERT,
            #                          avg_PERT_union_th_dice_array_C2_PERT]).T
            
            # df_temp = pd.DataFrame({'image': image,
            #                         'union_th_dice_array_C1_MC': tabella_temp[:, 0],
            #                         'union_th_dice_array_C2_MC': tabella_temp[:, 1],
            #                         'union_th_dice_array_C1_PERT': tabella_temp[:, 2],
            #                         'union_th_dice_array_C2_PERT': tabella_temp[:, 3]
            #                         })
            
            # if C1_exist:
            #     avg_PERT_dict_max_values_MC_C1['image_name'].append(image)
            #     avg_PERT_dict_max_values_MC_C1['union_max_dice'].append(np.nanmax(avg_PERT_union_th_dice_array_C1_MC))
            #     if not np.nanmax(avg_PERT_union_th_dice_array_C1_MC) >= 0: avg_PERT_dict_max_values_MC_C1['th_union_max_dice'].append(np.nan)
            #     else: avg_PERT_dict_max_values_MC_C1['th_union_max_dice'].append(th_range[np.where(avg_PERT_union_th_dice_array_C1_MC == np.nanmax(avg_PERT_union_th_dice_array_C1_MC))[0][0]])
                
            #     if np.max(avg_PERT_union_th_dice_array_C1_MC) > SI_dice_C1: 
            #         avg_PERT_dict_max_values_MC_C1['flag_union_dice'].append(1)
            #     else: 
            #         avg_PERT_dict_max_values_MC_C1['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(avg_PERT_dict_max_values_MC_C1)
            #     df_gen.to_csv(path_to_save_figure + "avg_PERT_DATAFRAME_max_values_MC_C1.csv", index=True)
            #     del df_gen
                
            #     avg_PERT_dict_max_values_PERT_C1['image_name'].append(image)
            #     avg_PERT_dict_max_values_PERT_C1['union_max_dice'].append(np.nanmax(avg_PERT_union_th_dice_array_C1_PERT))
            #     if not np.nanmax(avg_PERT_union_th_dice_array_C1_PERT) >= 0: avg_PERT_dict_max_values_PERT_C1['th_union_max_dice'].append(np.nan)
            #     else: avg_PERT_dict_max_values_PERT_C1['th_union_max_dice'].append(th_range[np.where(avg_PERT_union_th_dice_array_C1_PERT == np.nanmax(avg_PERT_union_th_dice_array_C1_PERT))[0][0]])
                
            #     if np.max(avg_PERT_union_th_dice_array_C1_PERT) > SI_dice_C1: 
            #         avg_PERT_dict_max_values_PERT_C1['flag_union_dice'].append(1)
            #     else: 
            #         avg_PERT_dict_max_values_PERT_C1['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(avg_PERT_dict_max_values_PERT_C1)
            #     df_gen.to_csv(path_to_save_figure + "avg_PERT_DATAFRAME_max_values_PERT_C1.csv", index=True)
            #     del df_gen
                
            #     title_C1 = "Classe 1"
            # else: title_C1 = "Classe 1 - Not present in avg_PERT"
                
            # if C2_exist:
                
            #     avg_PERT_dict_max_values_MC_C2['image_name'].append(image)
            #     avg_PERT_dict_max_values_MC_C2['union_max_dice'].append(np.nanmax(avg_PERT_union_th_dice_array_C2_MC))
            #     if not np.nanmax(avg_PERT_union_th_dice_array_C2_MC) >= 0: avg_PERT_dict_max_values_MC_C2['th_union_max_dice'].append(np.nan)
            #     else: avg_PERT_dict_max_values_MC_C2['th_union_max_dice'].append(th_range[np.where(avg_PERT_union_th_dice_array_C2_MC == np.nanmax(avg_PERT_union_th_dice_array_C2_MC))[0][0]])
                
            #     if np.max(avg_PERT_union_th_dice_array_C2_MC) > SI_dice_C2: 
            #         avg_PERT_dict_max_values_MC_C2['flag_union_dice'].append(2)
            #     else: 
            #         avg_PERT_dict_max_values_MC_C2['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(avg_PERT_dict_max_values_MC_C2)
            #     df_gen.to_csv(path_to_save_figure + "avg_PERT_DATAFRAME_max_values_MC_C2.csv", index=True)
            #     del df_gen
                    
            #     avg_PERT_dict_max_values_PERT_C2['image_name'].append(image)
            #     avg_PERT_dict_max_values_PERT_C2['union_max_dice'].append(np.nanmax(avg_PERT_union_th_dice_array_C2_PERT))
            #     if not np.nanmax(avg_PERT_union_th_dice_array_C2_PERT) >= 0: avg_PERT_dict_max_values_PERT_C2['th_union_max_dice'].append(np.nan)
            #     else: avg_PERT_dict_max_values_PERT_C2['th_union_max_dice'].append(th_range[np.where(avg_PERT_union_th_dice_array_C2_PERT == np.nanmax(avg_PERT_union_th_dice_array_C2_PERT))[0][0]])
                
            #     if np.max(avg_PERT_union_th_dice_array_C2_PERT) > SI_dice_C2: 
            #         avg_PERT_dict_max_values_PERT_C2['flag_union_dice'].append(2)
            #     else: 
            #         avg_PERT_dict_max_values_PERT_C2['flag_union_dice'].append(0)
                
            #     df_gen = pd.DataFrame.from_dict(avg_PERT_dict_max_values_PERT_C2)
            #     df_gen.to_csv(path_to_save_figure + "avg_PERT_DATAFRAME_max_values_PERT_C2.csv", index=True)
            #     del df_gen
            #     title_C2 = "Classe 2"
            # else: title_C2 = "Classe 2 - Not present in avg_PERT"
            
            
            # tabella_temp = np.array([avg_MC_union_th_dice_array_C1_MC,
            #                          avg_MC_union_th_dice_array_C2_MC,
            #                          avg_MC_union_th_dice_array_C1_PERT,
            #                          avg_MC_union_th_dice_array_C2_PERT]).T
            
            # df_temp = pd.DataFrame({'image': image,
            #                         'union_th_dice_array_C1_MC': tabella_temp[:, 0],
            #                         'union_th_dice_array_C2_MC': tabella_temp[:, 1],
            #                         'union_th_dice_array_C1_PERT': tabella_temp[:, 2],
            #                         'union_th_dice_array_C2_PERT': tabella_temp[:, 3]
            #                         })
            
            # if C1_exist:
            #     avg_MC_dict_max_values_MC_C1['image_name'].append(image)
            #     avg_MC_dict_max_values_MC_C1['union_max_dice'].append(np.nanmax(avg_MC_union_th_dice_array_C1_MC))
            #     if not np.nanmax(avg_MC_union_th_dice_array_C1_MC) >= 0: avg_MC_dict_max_values_MC_C1['th_union_max_dice'].append(np.nan)
            #     else: avg_MC_dict_max_values_MC_C1['th_union_max_dice'].append(th_range[np.where(avg_MC_union_th_dice_array_C1_MC == np.nanmax(avg_MC_union_th_dice_array_C1_MC))[0][0]])
                
            #     if np.max(avg_MC_union_th_dice_array_C1_MC) > SI_dice_C1: 
            #         avg_MC_dict_max_values_MC_C1['flag_union_dice'].append(1)
            #     else: 
            #         avg_MC_dict_max_values_MC_C1['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(avg_MC_dict_max_values_MC_C1)
            #     df_gen.to_csv(path_to_save_figure + "avg_MC_DATAFRAME_max_values_MC_C1.csv", index=True)
            #     del df_gen
                
            #     avg_MC_dict_max_values_PERT_C1['image_name'].append(image)
            #     avg_MC_dict_max_values_PERT_C1['union_max_dice'].append(np.nanmax(avg_MC_union_th_dice_array_C1_PERT))
            #     if not np.nanmax(avg_MC_union_th_dice_array_C1_PERT) >= 0: avg_MC_dict_max_values_PERT_C1['th_union_max_dice'].append(np.nan)
            #     else: avg_MC_dict_max_values_PERT_C1['th_union_max_dice'].append(th_range[np.where(avg_MC_union_th_dice_array_C1_PERT == np.nanmax(avg_MC_union_th_dice_array_C1_PERT))[0][0]])
                
            #     if np.max(avg_MC_union_th_dice_array_C1_PERT) > SI_dice_C1: 
            #         avg_MC_dict_max_values_PERT_C1['flag_union_dice'].append(1)
            #     else: 
            #         avg_MC_dict_max_values_PERT_C1['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(avg_MC_dict_max_values_PERT_C1)
            #     df_gen.to_csv(path_to_save_figure + "avg_MC_DATAFRAME_max_values_PERT_C1.csv", index=True)
            #     del df_gen
                
            #     title_C1 = "Classe 1"
            # else: title_C1 = "Classe 1 - Not present in avg_MC"
                
            # if C2_exist:
                
            #     avg_MC_dict_max_values_MC_C2['image_name'].append(image)
            #     avg_MC_dict_max_values_MC_C2['union_max_dice'].append(np.nanmax(avg_MC_union_th_dice_array_C2_MC))
            #     if not np.nanmax(avg_MC_union_th_dice_array_C2_MC) >= 0: avg_MC_dict_max_values_MC_C2['th_union_max_dice'].append(np.nan)
            #     else: avg_MC_dict_max_values_MC_C2['th_union_max_dice'].append(th_range[np.where(avg_MC_union_th_dice_array_C2_MC == np.nanmax(avg_MC_union_th_dice_array_C2_MC))[0][0]])
                
            #     if np.max(avg_MC_union_th_dice_array_C2_MC) > SI_dice_C2: 
            #         avg_MC_dict_max_values_MC_C2['flag_union_dice'].append(2)
            #     else: 
            #         avg_MC_dict_max_values_MC_C2['flag_union_dice'].append(0)
                    
            #     df_gen = pd.DataFrame.from_dict(avg_MC_dict_max_values_MC_C2)
            #     df_gen.to_csv(path_to_save_figure + "avg_MC_DATAFRAME_max_values_MC_C2.csv", index=True)
            #     del df_gen
                    
            #     avg_MC_dict_max_values_PERT_C2['image_name'].append(image)
            #     avg_MC_dict_max_values_PERT_C2['union_max_dice'].append(np.nanmax(avg_MC_union_th_dice_array_C2_PERT))
            #     if not np.nanmax(avg_MC_union_th_dice_array_C2_PERT) >= 0: avg_MC_dict_max_values_PERT_C2['th_union_max_dice'].append(np.nan)
            #     else: avg_MC_dict_max_values_PERT_C2['th_union_max_dice'].append(th_range[np.where(avg_MC_union_th_dice_array_C2_PERT == np.nanmax(avg_MC_union_th_dice_array_C2_PERT))[0][0]])
                
            #     if np.max(avg_MC_union_th_dice_array_C2_PERT) > SI_dice_C2: 
            #         avg_MC_dict_max_values_PERT_C2['flag_union_dice'].append(2)
            #     else: 
            #         avg_MC_dict_max_values_PERT_C2['flag_union_dice'].append(0)
                
            #     df_gen = pd.DataFrame.from_dict(avg_MC_dict_max_values_PERT_C2)
            #     df_gen.to_csv(path_to_save_figure + "avg_MC_DATAFRAME_max_values_PERT_C2.csv", index=True)
            #     del df_gen
            #     title_C2 = "Classe 2"
            # else: title_C2 = "Classe 2 - Not present in avg_MC"
            
            # colors_for_plot = [                               
            #     to_rgb('#006400'),
            #     to_rgb('#bc8f8f'),
            #     to_rgb('#ffd700'),
            #     to_rgb('#0000cd'),
            #     to_rgb('#00ff00'),
            #     to_rgb('#00ffff'),
            #     to_rgb('#1e90ff'),
            #     to_rgb('#ff1493')]
            
            # plt.figure(figsize=(30,15))
            # plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4])
            # plt.subplot(121)
            # plt.title(title_C1)
            # plt.plot(th_range,GT_union_th_dice_array_C1_MC, color=colors_for_plot[0], label="GT_Montecarlo")
            # plt.plot(th_range,GT_union_th_dice_array_C1_PERT, color=colors_for_plot[1], label="GT_Perturbation")
            # plt.plot(th_range,SI_union_th_dice_array_C1_MC, color=colors_for_plot[2], label="SI_Montecarlo")
            # plt.plot(th_range,SI_union_th_dice_array_C1_PERT, color=colors_for_plot[3], label="SI_Perturbation")
            # plt.plot(th_range,avg_PERT_union_th_dice_array_C1_MC, color=colors_for_plot[4], label="avg_PERT_Montecarlo")
            # plt.plot(th_range,avg_PERT_union_th_dice_array_C1_PERT, color=colors_for_plot[5], label="avg_PERT_Perturbation")
            # plt.plot(th_range,avg_MC_union_th_dice_array_C1_MC, color=colors_for_plot[6], label="avg_MC_Montecarlo")
            # plt.plot(th_range,avg_MC_union_th_dice_array_C1_PERT, color=colors_for_plot[7], label="avg_MC_Perturbation")
            # plt.axhline(SI_dice_C1, label="Baseline SI vs GT", color='r', linestyle='--')
            # plt.legend(loc=4)
            # plt.subplot(122)
            # plt.title(title_C2)
            # plt.plot(th_range,GT_union_th_dice_array_C2_MC, color=colors_for_plot[0], label="GT_Montecarlo")
            # plt.plot(th_range,GT_union_th_dice_array_C2_PERT, color=colors_for_plot[1], label="GT_Perturbation")
            # plt.plot(th_range,SI_union_th_dice_array_C2_MC, color=colors_for_plot[2], label="SI_Montecarlo")
            # plt.plot(th_range,SI_union_th_dice_array_C2_PERT, color=colors_for_plot[3], label="SI_Perturbation")
            # plt.plot(th_range,avg_PERT_union_th_dice_array_C2_MC, color=colors_for_plot[4], label="avg_PERT_Montecarlo")
            # plt.plot(th_range,avg_PERT_union_th_dice_array_C2_PERT, color=colors_for_plot[5], label="avg_PERT_Perturbation")
            # plt.plot(th_range,avg_MC_union_th_dice_array_C2_MC, color=colors_for_plot[6], label="avg_MC_Montecarlo")
            # plt.plot(th_range,avg_MC_union_th_dice_array_C2_PERT, color=colors_for_plot[7], label="avg_MC_Perturbation")
            # plt.axhline(SI_dice_C2, label="Baseline SI vs GT", color='r', linestyle='--')
            # plt.legend(loc=4)
            # plt.savefig(path_to_save_figure + image[:-4] + "_dice_x_threshold_MC_or_PERT.png")
            # plt.close()
            
            GT_max_C1 = np.max(np.array([GT_union_th_dice_array_C1_MC,GT_union_th_dice_array_C1_PERT]).T,axis=-1)
            GT_max_C2 = np.max(np.array([GT_union_th_dice_array_C2_MC,GT_union_th_dice_array_C2_PERT]).T,axis=-1)
            # SI_max_C1 = np.max(np.array([SI_union_th_dice_array_C1_MC,SI_union_th_dice_array_C1_PERT]).T,axis=-1)
            # SI_max_C2 = np.max(np.array([SI_union_th_dice_array_C2_MC,SI_union_th_dice_array_C2_PERT]).T,axis=-1)
            # avg_PERT_max_C1 = np.max(np.array([avg_PERT_union_th_dice_array_C1_MC,avg_PERT_union_th_dice_array_C1_PERT]).T,axis=-1)
            # avg_PERT_max_C2 = np.max(np.array([avg_PERT_union_th_dice_array_C2_MC,avg_PERT_union_th_dice_array_C2_PERT]).T,axis=-1)
            # avg_MC_max_C1 = np.max(np.array([avg_MC_union_th_dice_array_C1_MC,avg_MC_union_th_dice_array_C1_PERT]).T,axis=-1)
            # avg_MC_max_C2 = np.max(np.array([avg_MC_union_th_dice_array_C2_MC,avg_MC_union_th_dice_array_C2_PERT]).T,axis=-1)

            # plt.figure(figsize=(30,15))
            # plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4])
            # plt.subplot(121)
            # plt.title(title_C1)
            # plt.plot(th_range,GT_max_C1, color=colors_for_plot[0], label="GT - Max between MC and PERT")
            # plt.plot(th_range,SI_max_C1, color=colors_for_plot[1], label="SI - Max between MC and PERT")
            # plt.plot(th_range,avg_PERT_max_C1, color=colors_for_plot[2], label="avg_PERT - Max between MC and PERT")
            # plt.plot(th_range,avg_MC_max_C1, color=colors_for_plot[3], label="avg_MC - Max between MC and PERT")
            # plt.axhline(SI_dice_C1, label="Baseline SI vs GT", color='r', linestyle='--')
            # plt.legend(loc=4)
            # plt.subplot(122)
            # plt.title(title_C2)
            # plt.plot(th_range,GT_max_C2, color=colors_for_plot[0], label="GT - Max between MC and PERT")
            # plt.plot(th_range,SI_max_C2, color=colors_for_plot[1], label="SI - Max between MC and PERT")
            # plt.plot(th_range,avg_PERT_max_C2, color=colors_for_plot[2], label="avg_PERT - Max between MC and PERT")
            # plt.plot(th_range,avg_MC_max_C2, color=colors_for_plot[3], label="avg_MC - Max between MC and PERT")
            # plt.axhline(SI_dice_C2, label="Baseline SI vs GT", color='r', linestyle='--')
            # plt.legend(loc=4)
            # plt.savefig(path_to_save_figure + image[:-4] + "_dice_x_threshold_max_MC_PERT.png")
            # plt.close()
            
            
            GT_SI_dice_array_C1.append(wjb_functions.dice(GT_mask_C1,SI_mask_C1))
            GT_SI_dice_array_C2.append(wjb_functions.dice(GT_mask_C2,SI_mask_C2))
            
            
            GT_monotony_MC_C1 = wjb_functions.pearsons(th_range,GT_union_th_dice_array_C1_MC)[0][1]
            GT_monotony_MC_C2 = wjb_functions.pearsons(th_range,GT_union_th_dice_array_C2_MC)[0][1]
            GT_monotony_PERT_C1 = wjb_functions.pearsons(th_range,GT_union_th_dice_array_C1_PERT)[0][1]
            GT_monotony_PERT_C2 = wjb_functions.pearsons(th_range,GT_union_th_dice_array_C2_PERT)[0][1]
            
            GT_monotony_MC_C1_array.append(GT_monotony_MC_C1)
            GT_monotony_MC_C2_array.append(GT_monotony_MC_C2)
            GT_monotony_PERT_C1_array.append(GT_monotony_PERT_C1)
            GT_monotony_PERT_C2_array.append(GT_monotony_PERT_C2)
            
            # SI_monotony_MC_C1 = wjb_functions.pearsons(th_range,SI_union_th_dice_array_C1_MC)[0][1]
            # SI_monotony_MC_C2 = wjb_functions.pearsons(th_range,SI_union_th_dice_array_C2_MC)[0][1]
            # SI_monotony_PERT_C1 = wjb_functions.pearsons(th_range,SI_union_th_dice_array_C1_PERT)[0][1]
            # SI_monotony_PERT_C2 = wjb_functions.pearsons(th_range,SI_union_th_dice_array_C2_PERT)[0][1]
            
            # SI_monotony_MC_C1_array.append(SI_monotony_MC_C1)
            # SI_monotony_MC_C2_array.append(SI_monotony_MC_C2)
            # SI_monotony_PERT_C1_array.append(SI_monotony_PERT_C1)
            # SI_monotony_PERT_C2_array.append(SI_monotony_PERT_C2)
            
            # avg_PERT_monotony_MC_C1 = wjb_functions.pearsons(th_range,avg_PERT_union_th_dice_array_C1_MC)[0][1]
            # avg_PERT_monotony_MC_C2 = wjb_functions.pearsons(th_range,avg_PERT_union_th_dice_array_C2_MC)[0][1]
            # avg_PERT_monotony_PERT_C1 = wjb_functions.pearsons(th_range,avg_PERT_union_th_dice_array_C1_PERT)[0][1]
            # avg_PERT_monotony_PERT_C2 = wjb_functions.pearsons(th_range,avg_PERT_union_th_dice_array_C2_PERT)[0][1]
            
            # avg_PERT_monotony_MC_C1_array.append(avg_PERT_monotony_MC_C1)
            # avg_PERT_monotony_MC_C2_array.append(avg_PERT_monotony_MC_C2)
            # avg_PERT_monotony_PERT_C1_array.append(avg_PERT_monotony_PERT_C1)
            # avg_PERT_monotony_PERT_C2_array.append(avg_PERT_monotony_PERT_C2)
            
            # avg_MC_monotony_MC_C1 = wjb_functions.pearsons(th_range,avg_MC_union_th_dice_array_C1_MC)[0][1]
            # avg_MC_monotony_MC_C2 = wjb_functions.pearsons(th_range,avg_MC_union_th_dice_array_C2_MC)[0][1]
            # avg_MC_monotony_PERT_C1 = wjb_functions.pearsons(th_range,avg_MC_union_th_dice_array_C1_PERT)[0][1]
            # avg_MC_monotony_PERT_C2 = wjb_functions.pearsons(th_range,avg_MC_union_th_dice_array_C2_PERT)[0][1]
            
            # avg_MC_monotony_MC_C1_array.append(avg_MC_monotony_MC_C1)
            # avg_MC_monotony_MC_C2_array.append(avg_MC_monotony_MC_C2)
            # avg_MC_monotony_PERT_C1_array.append(avg_MC_monotony_PERT_C1)
            # avg_MC_monotony_PERT_C2_array.append(avg_MC_monotony_PERT_C2)
            
            GT_monotony_max_MC_PERT_C1 = wjb_functions.pearsons(th_range,GT_max_C1)[0][1]
            GT_monotony_max_MC_PERT_C2 = wjb_functions.pearsons(th_range,GT_max_C2)[0][1]
            
            GT_monotony_max_MC_PERT_C1_array.append(GT_monotony_max_MC_PERT_C1)
            GT_monotony_max_MC_PERT_C2_array.append(GT_monotony_max_MC_PERT_C2)
            
            # SI_monotony_max_MC_PERT_C1 = wjb_functions.pearsons(th_range,SI_max_C1)[0][1]
            # SI_monotony_max_MC_PERT_C2 = wjb_functions.pearsons(th_range,SI_max_C2)[0][1]
            
            # SI_monotony_max_MC_PERT_C1_array.append(SI_monotony_max_MC_PERT_C1)
            # SI_monotony_max_MC_PERT_C2_array.append(SI_monotony_max_MC_PERT_C2)
            
            # avg_PERT_monotony_max_MC_PERT_C1 = wjb_functions.pearsons(th_range,avg_PERT_max_C1)[0][1]
            # avg_PERT_monotony_max_MC_PERT_C2 = wjb_functions.pearsons(th_range,avg_PERT_max_C2)[0][1]
            
            # avg_PERT_monotony_max_MC_PERT_C1_array.append(avg_PERT_monotony_max_MC_PERT_C1)
            # avg_PERT_monotony_max_MC_PERT_C2_array.append(avg_PERT_monotony_max_MC_PERT_C2)
            
            # avg_MC_monotony_max_MC_PERT_C1 = wjb_functions.pearsons(th_range,avg_MC_max_C1)[0][1]
            # avg_MC_monotony_max_MC_PERT_C2 = wjb_functions.pearsons(th_range,avg_MC_max_C2)[0][1]
            
            # avg_MC_monotony_max_MC_PERT_C1_array.append(avg_MC_monotony_max_MC_PERT_C1)
            # avg_MC_monotony_max_MC_PERT_C2_array.append(avg_MC_monotony_max_MC_PERT_C2)
            
            # chiude ciclo per ogni immagine
            
            SI_UM_dice_MC_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1) - wjb_functions.dice(GT_mask_C1,mask_union_C1_MC)
            SI_UM_dice_MC_C1_array.append(SI_UM_dice_MC_C1)
            
            SI_UM_max_dice_MC_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1) - wjb_functions.dice(GT_mask_C1,UM_max_dice_MC_C1)
            SI_UM_max_dice_MC_C1_array.append(SI_UM_max_dice_MC_C1)
            
            SI_UM_dice_MC_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2) - wjb_functions.dice(GT_mask_C2,mask_union_C2_MC)
            SI_UM_dice_MC_C2_array.append(SI_UM_dice_MC_C2)
            
            SI_UM_max_dice_MC_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2) - wjb_functions.dice(GT_mask_C2,UM_max_dice_MC_C2)
            SI_UM_max_dice_MC_C2_array.append(SI_UM_max_dice_MC_C2)
            
            SI_UM_dice_PERT_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1) - wjb_functions.dice(GT_mask_C1,mask_union_C1_PERT)
            SI_UM_dice_PERT_C1_array.append(SI_UM_dice_PERT_C1)
            
            SI_UM_max_dice_PERT_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1) - wjb_functions.dice(GT_mask_C1,UM_max_dice_PERT_C1)
            SI_UM_max_dice_PERT_C1_array.append(SI_UM_max_dice_PERT_C1)
            
            SI_UM_dice_PERT_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2) - wjb_functions.dice(GT_mask_C2,mask_union_C2_PERT)
            SI_UM_dice_PERT_C2_array.append(SI_UM_dice_PERT_C2)
            
            SI_UM_max_dice_PERT_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2) - wjb_functions.dice(GT_mask_C2,UM_max_dice_PERT_C2)
            SI_UM_max_dice_PERT_C2_array.append(SI_UM_max_dice_PERT_C2)
        
        
        plt.figure(figsize=(16,15))
        plt.suptitle("Difference between SI_dice and UM_dice, dataset: " + dataset + ", subset: " + subset + ", Montecarlo")
        plt.subplot(221)
        rho = wjb_functions.pearsons(GT_monotony_MC_C1_array, SI_UM_dice_MC_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_MC_C1_array,SI_UM_dice_MC_C1_array, color=colors_for_plot[0], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(222)
        rho = wjb_functions.pearsons(GT_monotony_MC_C1_array, SI_UM_dice_PERT_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_MC_C1_array,SI_UM_dice_PERT_C1_array, color=colors_for_plot[0], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(223)
        rho = wjb_functions.pearsons(GT_monotony_MC_C2_array, SI_UM_dice_MC_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 2, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_MC_C2_array,SI_UM_dice_MC_C2_array, color=colors_for_plot[0], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(224)
        rho = wjb_functions.pearsons(GT_monotony_MC_C2_array, SI_UM_dice_PERT_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 2, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_MC_C2_array,SI_UM_dice_PERT_C2_array, color=colors_for_plot[0], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.savefig(path_to_save_figure + "correlation_between_Monotony_of_GT_dice_curve_with_UM_AND_difference_in_dices_GTSI_or_GTUM_montecarlo.png")
        plt.close()
        
        plt.figure(figsize=(16,15))
        plt.suptitle("Difference between SI_dice and UM_dice, dataset: " + dataset + ", subset: " + subset + ", Perturbations")
        plt.subplot(221)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C1_array, SI_UM_dice_MC_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C1_array,SI_UM_dice_MC_C1_array, color=colors_for_plot[1], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(222)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C1_array, SI_UM_dice_PERT_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C1_array,SI_UM_dice_PERT_C1_array, color=colors_for_plot[1], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(223)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C2_array, SI_UM_dice_MC_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 2, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C2_array,SI_UM_dice_MC_C2_array, color=colors_for_plot[1], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(224)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C2_array, SI_UM_dice_PERT_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 2, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C2_array,SI_UM_dice_PERT_C2_array, color=colors_for_plot[1], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.savefig(path_to_save_figure + "correlation_between_Monotony_of_GT_dice_curve_with_UM_AND_difference_in_dices_GTSI_or_GTUM_perturbation.png")
        plt.close()
        
        
        
        plt.figure(figsize=(16,15))
        plt.suptitle("Difference between SI_dice and UM_max_dice, dataset: " + dataset + ", subset: " + subset + ", Montecarlo")
        plt.subplot(221)
        rho = wjb_functions.pearsons(GT_monotony_MC_C1_array, SI_UM_max_dice_MC_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_MC_C1_array,SI_UM_max_dice_MC_C1_array, color=colors_for_plot[0], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(222)
        rho = wjb_functions.pearsons(GT_monotony_MC_C1_array, SI_UM_max_dice_PERT_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C1_array,SI_UM_max_dice_PERT_C1_array, color=colors_for_plot[0], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(223)
        rho = wjb_functions.pearsons(GT_monotony_MC_C2_array, SI_UM_max_dice_MC_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 2")
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_MC_C2_array,SI_UM_max_dice_MC_C2_array, color=colors_for_plot[0], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(224)
        rho = wjb_functions.pearsons(GT_monotony_MC_C2_array, SI_UM_max_dice_PERT_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 2, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C2_array,SI_UM_max_dice_PERT_C2_array, color=colors_for_plot[0], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.savefig(path_to_save_figure + "correlation_between_Monotony_of_GT_max_dice_curve_with_UM_AND_difference_in_max_dices_GTSI_or_GTUM_montecarlo.png")
        plt.close()
        
        plt.figure(figsize=(16,15))
        plt.suptitle("Difference between SI_dice and UM_max_dice, dataset: " + dataset + ", subset: " + subset + ", Perturbations")
        plt.subplot(221)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C1_array, SI_UM_max_dice_MC_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C1_array,SI_UM_max_dice_MC_C1_array, color=colors_for_plot[1], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(222)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C1_array, SI_UM_max_dice_PERT_C1_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 1, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C1_array,SI_UM_max_dice_PERT_C1_array, color=colors_for_plot[1], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(223)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C2_array, SI_UM_max_dice_MC_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_MC x GT - Class 2, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C2_array,SI_UM_max_dice_MC_C2_array, color=colors_for_plot[1], label='Union Mask MC')
        plt.xlabel("Monotony of dice curve UM_MC x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.subplot(224)
        rho = wjb_functions.pearsons(GT_monotony_PERT_C2_array, SI_UM_max_dice_PERT_C2_array)[0][1]
        rho_str = str(np.round(rho,3))
        plt.title("Monotony - UM_PERT x GT - Class 2, rho = " + rho_str)
        plt.axhline(0,linestyle='--',color='r',label="Above this line, SI is better")
        plt.scatter(GT_monotony_PERT_C2_array,SI_UM_max_dice_PERT_C2_array, color=colors_for_plot[1], label='Union Mask PERT')
        plt.xlabel("Monotony of dice curve UM_PERT x GT")
        plt.ylabel("Difference between dices GT_SI and GT_UM")
        plt.legend()
        plt.savefig(path_to_save_figure + "correlation_between_Monotony_of_GT_max_dice_curve_with_UM_AND_difference_in_max_dices_GTSI_or_GTUM_perturbation.png")
        plt.close()
        
        
        # plt.figure(figsize=(17,8))
        # plt.suptitle("Monotony of unionmask dice with specific mask as 'GT' - dataset: " + dataset[:-1] + " , subset: " + subset)
        # plt.subplot(121)
        # plt.title(title_C1)
        # plt.plot(GT_monotony_MC_C1_array, color=colors_for_plot[0], label='GT_MC')
        # plt.plot(SI_monotony_MC_C1_array, color=colors_for_plot[1], label='SI_MC')
        # plt.plot(avg_PERT_monotony_MC_C1_array, color=colors_for_plot[2], label='avg_PERT_MC')
        # plt.plot(avg_MC_monotony_MC_C1_array, color=colors_for_plot[3], label='avg_MC_MC')
        # plt.plot(GT_monotony_PERT_C1_array, color=colors_for_plot[4], label='GT_PERT')
        # plt.plot(SI_monotony_PERT_C1_array, color=colors_for_plot[5], label='SI_PERT')
        # plt.plot(avg_PERT_monotony_PERT_C1_array, color=colors_for_plot[6], label='avg_PERT_PERT')
        # plt.plot(avg_MC_monotony_PERT_C1_array, color=colors_for_plot[7], label='avg_MC_PERT')
        # plt.legend()
        # plt.subplot(122)
        # plt.title(title_C2)
        # plt.plot(GT_monotony_MC_C2_array, color=colors_for_plot[0], label='GT_MC')
        # plt.plot(SI_monotony_MC_C2_array, color=colors_for_plot[1], label='SI_MC')
        # plt.plot(avg_PERT_monotony_MC_C2_array, color=colors_for_plot[2], label='avg_PERT_MC')
        # plt.plot(avg_MC_monotony_MC_C2_array, color=colors_for_plot[3], label='avg_MC_MC')
        # plt.plot(GT_monotony_PERT_C2_array, color=colors_for_plot[4], label='GT_PERT')
        # plt.plot(SI_monotony_PERT_C2_array, color=colors_for_plot[5], label='SI_PERT')
        # plt.plot(avg_PERT_monotony_PERT_C2_array, color=colors_for_plot[6], label='avg_PERT_PERT')
        # plt.plot(avg_MC_monotony_PERT_C2_array, color=colors_for_plot[7], label='avg_MC_PERT')
        # plt.legend()
        # plt.savefig(path_to_save_figure + "Monotony_of_unionmask_dice_with_different_masks_used_as_GT.png")
        # plt.close()
        
        # scatterx = np.arange(len(GT_monotony_max_MC_PERT_C1_array))+1
        
        # plt.figure(figsize=(17,8))
        # plt.suptitle("Monotony of unionmask dice with specific mask as 'GT' - dataset: " + dataset[:-1] + " , subset: " + subset)
        # plt.subplot(121)
        # plt.title(title_C1)
        # plt.plot(scatterx,GT_monotony_max_MC_PERT_C1_array, color=colors_for_plot[0], label='GT_max_MC_PERT')
        # plt.plot(scatterx,SI_monotony_max_MC_PERT_C1_array, color=colors_for_plot[1], label='SI_max_MC_PERT')
        # plt.plot(scatterx,avg_PERT_monotony_max_MC_PERT_C1_array, color=colors_for_plot[2], label='avg_PERT_max_MC_PERT')
        # plt.plot(scatterx,avg_MC_monotony_max_MC_PERT_C1_array, color=colors_for_plot[3], label='avg_MC_max_MC_PERT')
        # plt.legend()
        # plt.subplot(122)
        # plt.title(title_C2)
        # plt.plot(scatterx,GT_monotony_max_MC_PERT_C2_array, color=colors_for_plot[0], label='GT_max_MC_PERT')
        # plt.plot(scatterx,SI_monotony_max_MC_PERT_C2_array, color=colors_for_plot[1], label='SI_max_MC_PERT')
        # plt.plot(scatterx,avg_PERT_monotony_max_MC_PERT_C2_array, color=colors_for_plot[2], label='avg_PERT_max_MC_PERT')
        # plt.plot(scatterx,avg_MC_monotony_max_MC_PERT_C2_array, color=colors_for_plot[3], label='avg_MC_max_MC_PERT')
        # plt.legend()
        # plt.savefig(path_to_save_figure + "Monotony_of_unionmask_dice_with_different_masks_used_as_GT_USING_MAX_MC_PERT.png")
        
        
        
        # rho_GT_SI_MC_C1 = wjb_functions.pearsons(GT_monotony_MC_C1_array,SI_monotony_MC_C1_array)[0][1]
        # rho_GT_SI_PERT_C1 = wjb_functions.pearsons(GT_monotony_PERT_C1_array,SI_monotony_PERT_C1_array)[0][1]
        # rho_GT_avg_PERT_MC_C1 = wjb_functions.pearsons(GT_monotony_MC_C1_array,avg_PERT_monotony_MC_C1_array)[0][1]
        # rho_GT_avg_PERT_PERT_C1 = wjb_functions.pearsons(GT_monotony_PERT_C1_array,avg_PERT_monotony_PERT_C1_array)[0][1]
        # rho_GT_avg_MC_MC_C1 = wjb_functions.pearsons(GT_monotony_MC_C1_array,avg_MC_monotony_MC_C1_array)[0][1]
        # rho_GT_avg_MC_PERT_C1 = wjb_functions.pearsons(GT_monotony_PERT_C1_array,avg_MC_monotony_PERT_C1_array)[0][1]
        
        # rho_GT_SI_MC_C2 = wjb_functions.pearsons(GT_monotony_MC_C2_array,SI_monotony_MC_C2_array)[0][1]
        # rho_GT_SI_PERT_C2 = wjb_functions.pearsons(GT_monotony_PERT_C2_array,SI_monotony_PERT_C2_array)[0][1]
        # rho_GT_avg_PERT_MC_C2 = wjb_functions.pearsons(GT_monotony_MC_C2_array,avg_PERT_monotony_MC_C2_array)[0][1]
        # rho_GT_avg_PERT_PERT_C2 = wjb_functions.pearsons(GT_monotony_PERT_C2_array,avg_PERT_monotony_PERT_C2_array)[0][1]
        # rho_GT_avg_MC_MC_C2 = wjb_functions.pearsons(GT_monotony_MC_C2_array,avg_MC_monotony_MC_C2_array)[0][1]
        # rho_GT_avg_MC_PERT_C2 = wjb_functions.pearsons(GT_monotony_PERT_C2_array,avg_MC_monotony_PERT_C2_array)[0][1]
        
        # rho_GT_SI_max_MC_PERT_C1 = wjb_functions.pearsons(GT_monotony_max_MC_PERT_C1_array,SI_monotony_max_MC_PERT_C1_array)[0][1]
        # rho_GT_avg_PERT_max_MC_PERT_C1 = wjb_functions.pearsons(GT_monotony_max_MC_PERT_C1_array,avg_PERT_monotony_max_MC_PERT_C1_array)[0][1]
        # rho_GT_avg_MC_max_MC_PERT_C1 = wjb_functions.pearsons(GT_monotony_max_MC_PERT_C1_array,avg_MC_monotony_max_MC_PERT_C1_array)[0][1]
        
        # rho_GT_SI_max_MC_PERT_C2 = wjb_functions.pearsons(GT_monotony_max_MC_PERT_C2_array,SI_monotony_max_MC_PERT_C2_array)[0][1]
        # rho_GT_avg_PERT_max_MC_PERT_C2 = wjb_functions.pearsons(GT_monotony_max_MC_PERT_C2_array,avg_PERT_monotony_max_MC_PERT_C2_array)[0][1]
        # rho_GT_avg_MC_max_MC_PERT_C2 = wjb_functions.pearsons(GT_monotony_max_MC_PERT_C2_array,avg_MC_monotony_max_MC_PERT_C2_array)[0][1]
        
        # rho_GT_SI_MC_C1_array.append(rho_GT_SI_MC_C1)
        # rho_GT_SI_PERT_C1_array.append(rho_GT_SI_PERT_C1)
        # rho_GT_avg_PERT_MC_C1_array.append(rho_GT_avg_PERT_MC_C1)
        # rho_GT_avg_PERT_PERT_C1_array.append(rho_GT_avg_PERT_PERT_C1)
        # rho_GT_avg_MC_MC_C1_array.append(rho_GT_avg_MC_MC_C1)
        # rho_GT_avg_MC_PERT_C1_array.append(rho_GT_avg_MC_PERT_C1)
        
        # rho_GT_SI_MC_C2_array.append(rho_GT_SI_MC_C2)
        # rho_GT_SI_PERT_C2_array.append(rho_GT_SI_PERT_C2)
        # rho_GT_avg_PERT_MC_C2_array.append(rho_GT_avg_PERT_MC_C2)
        # rho_GT_avg_PERT_PERT_C2_array.append(rho_GT_avg_PERT_PERT_C2)
        # rho_GT_avg_MC_MC_C2_array.append(rho_GT_avg_MC_MC_C2)
        # rho_GT_avg_MC_PERT_C2_array.append(rho_GT_avg_MC_PERT_C2)
        
        # rho_GT_SI_max_MC_PERT_C1_array.append(rho_GT_SI_max_MC_PERT_C1)
        # rho_GT_avg_PERT_max_MC_PERT_C1_array.append(rho_GT_avg_PERT_max_MC_PERT_C1)
        # rho_GT_avg_MC_max_MC_PERT_C1_array.append(rho_GT_avg_MC_max_MC_PERT_C1)
        
        # rho_GT_SI_max_MC_PERT_C2_array.append(rho_GT_SI_max_MC_PERT_C2)
        # rho_GT_avg_PERT_max_MC_PERT_C2_array.append(rho_GT_avg_PERT_max_MC_PERT_C2)
        # rho_GT_avg_MC_max_MC_PERT_C2_array.append(rho_GT_avg_MC_max_MC_PERT_C2)
        
        
        # corr_dict_C1 = {
        #     'Corr_GT_SI_MC_C1': rho_GT_SI_MC_C1_array,
        #     'Corr_GT_SI_PERT_C1': rho_GT_SI_PERT_C1_array,
        #     'Corr_GT_avg_PERT_MC_C1': rho_GT_avg_PERT_MC_C1_array,
        #     'Corr_GT_avg_PERT_PERT_C1': rho_GT_avg_PERT_PERT_C1_array,
        #     'Corr_GT_avg_MC_MC_C1': rho_GT_avg_MC_MC_C1_array,
        #     'Corr_GT_avg_MC_PERT_C1': rho_GT_avg_MC_PERT_C1_array,
        #     }
        
        # corr_dict_C2 = {
        #     'Corr_GT_SI_MC_C2': rho_GT_SI_MC_C2_array,
        #     'Corr_GT_SI_PERT_C2': rho_GT_SI_PERT_C2_array,
        #     'Corr_GT_avg_PERT_MC_C2': rho_GT_avg_PERT_MC_C2_array,
        #     'Corr_GT_avg_PERT_PERT_C2': rho_GT_avg_PERT_PERT_C2_array,
        #     'Corr_GT_avg_MC_MC_C2': rho_GT_avg_MC_MC_C2_array,
        #     'Corr_GT_avg_MC_PERT_C2': rho_GT_avg_MC_PERT_C2_array,
        #     }
        
        # df_gen_C1 = pd.DataFrame.from_dict(corr_dict_C1)
        # df_gen_C1.to_csv(general_path_to_save + dataset_name + subset + "_correlations_between_monotonies_of_UM_with_GT_and_UM_with_other_masks_C1.csv", index=False)
        # df_gen_C2 = pd.DataFrame.from_dict(corr_dict_C2)
        # df_gen_C2.to_csv(general_path_to_save + dataset_name + subset + "_correlations_between_monotonies_of_UM_with_GT_and_UM_with_other_masks_C2.csv", index=False)
        
        # print(dataset + subset + ", Correlations C1: ")
        # print(corr_dict_C1)
        # print(dataset + subset + ", Correlations C2: ")
        # print(corr_dict_C2)
        # # plt.figure(figsize=(26,16))
        # # plt.suptitle("Class 1, Correlations between monotonies: unionmask-GT vs unionmask-mask - dataset: " + dataset + ", subset: " + subset)
        # # plt.scatter(rho_GT_SI_MC_C1_array, color=colors_for_plot[0], label='SI - MC')
        # # plt.scatter(rho_GT_SI_PERT_C1_array, color=colors_for_plot[1], label='SI - PERT')
        # # plt.scatter(rho_GT_avg_PERT_MC_C1_array, color=colors_for_plot[0], label='avg_PERT - MC')
        # # plt.scatter(rho_GT_avg_PERT_PERT_C1_array, color=colors_for_plot[1], label='avg_PERT - PERT')
        # # plt.scatter(rho_GT_avg_MC_MC_C1_array, color=colors_for_plot[0], label='avg_MC - MC')
        # # plt.scatter(rho_GT_avg_MC_PERT_C1_array, color=colors_for_plot[1], label='avg_MC - PERT')
        # # plt.legend()
        # # plt.savefig()
        
        # #%%
        # dice_rho_GT_monotony_MC_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, GT_monotony_MC_C1_array)[0][1]
        # dice_rho_SI_monotony_MC_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, SI_monotony_MC_C1_array)[0][1]
        # dice_rho_avg_PERT_monotony_MC_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, avg_PERT_monotony_MC_C1_array)[0][1]
        # dice_rho_avg_MC_monotony_MC_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, avg_MC_monotony_MC_C1_array)[0][1]
        
        # dice_rho_GT_monotony_MC_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, GT_monotony_MC_C2_array)[0][1]
        # dice_rho_SI_monotony_MC_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, SI_monotony_MC_C2_array)[0][1]
        # dice_rho_avg_PERT_monotony_MC_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, avg_PERT_monotony_MC_C2_array)[0][1]
        # dice_rho_avg_MC_monotony_MC_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, avg_MC_monotony_MC_C2_array)[0][1]
        
        # dice_rho_GT_monotony_PERT_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, GT_monotony_PERT_C1_array)[0][1]
        # dice_rho_SI_monotony_PERT_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, SI_monotony_PERT_C1_array)[0][1]
        # dice_rho_avg_PERT_monotony_PERT_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, avg_PERT_monotony_PERT_C1_array)[0][1]
        # dice_rho_avg_MC_monotony_PERT_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, avg_MC_monotony_PERT_C1_array)[0][1]
        
        # dice_rho_GT_monotony_PERT_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, GT_monotony_PERT_C2_array)[0][1]
        # dice_rho_SI_monotony_PERT_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, SI_monotony_PERT_C2_array)[0][1]
        # dice_rho_avg_PERT_monotony_PERT_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, avg_PERT_monotony_PERT_C2_array)[0][1]
        # dice_rho_avg_MC_monotony_PERT_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, avg_MC_monotony_PERT_C2_array)[0][1]
        
        
        # dice_rho_GT_monotony_max_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, GT_monotony_max_MC_PERT_C1_array)[0][1]
        # dice_rho_GT_monotony_max_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, GT_monotony_max_MC_PERT_C2_array)[0][1]
        
        # dice_rho_SI_monotony_max_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, SI_monotony_max_MC_PERT_C1_array)[0][1]
        # dice_rho_SI_monotony_max_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, SI_monotony_max_MC_PERT_C2_array)[0][1]
        
        # dice_rho_avg_PERT_monotony_max_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, avg_PERT_monotony_max_MC_PERT_C1_array)[0][1]
        # dice_rho_avg_PERT_monotony_max_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, avg_PERT_monotony_max_MC_PERT_C2_array)[0][1]
        
        # dice_rho_avg_MC_monotony_max_C1 = wjb_functions.pearsons(GT_SI_dice_array_C1, avg_MC_monotony_max_MC_PERT_C1_array)[0][1]
        # dice_rho_avg_MC_monotony_max_C2 = wjb_functions.pearsons(GT_SI_dice_array_C2, avg_MC_monotony_max_MC_PERT_C2_array)[0][1]
        
        # dict_rho_dices_vs_monotonies = {
        # "dice_rho_GT_monotony_MC_C1": [dice_rho_GT_monotony_MC_C1],
        # "dice_rho_SI_monotony_MC_C1": [dice_rho_SI_monotony_MC_C1],
        # "dice_rho_avg_PERT_monotony_MC_C1": [dice_rho_avg_PERT_monotony_MC_C1],
        # "dice_rho_avg_MC_monotony_MC_C1": [dice_rho_avg_MC_monotony_MC_C1],
        
        # "dice_rho_GT_monotony_MC_C2": [dice_rho_GT_monotony_MC_C2],
        # "dice_rho_SI_monotony_MC_C2": [dice_rho_SI_monotony_MC_C2],
        # "dice_rho_avg_PERT_monotony_MC_C2": [dice_rho_avg_PERT_monotony_MC_C2],
        # "dice_rho_avg_MC_monotony_MC_C2": [dice_rho_avg_MC_monotony_MC_C2],
        
        # "dice_rho_GT_monotony_PERT_C1": [dice_rho_GT_monotony_PERT_C1],
        # "dice_rho_SI_monotony_PERT_C1": [dice_rho_SI_monotony_PERT_C1],
        # "dice_rho_avg_PERT_monotony_PERT_C1": [dice_rho_avg_PERT_monotony_PERT_C1],
        # "dice_rho_avg_MC_monotony_PERT_C1": [dice_rho_avg_MC_monotony_PERT_C1],
        
        # "dice_rho_GT_monotony_PERT_C2": [dice_rho_GT_monotony_PERT_C2],
        # "dice_rho_SI_monotony_PERT_C2": [dice_rho_SI_monotony_PERT_C2],
        # "dice_rho_avg_PERT_monotony_PERT_C2": [dice_rho_avg_PERT_monotony_PERT_C2],
        # "dice_rho_avg_MC_monotony_PERT_C2": [dice_rho_avg_MC_monotony_PERT_C2],
        
        # "dice_rho_GT_monotony_max_C1": [dice_rho_GT_monotony_max_C1],
        # "dice_rho_GT_monotony_max_C2": [dice_rho_GT_monotony_max_C2],
        
        # "dice_rho_SI_monotony_max_C1": [dice_rho_SI_monotony_max_C1],
        # "dice_rho_SI_monotony_max_C2": [dice_rho_SI_monotony_max_C2],
        
        # "dice_rho_avg_PERT_monotony_max_C1": [dice_rho_avg_PERT_monotony_max_C1],
        # "dice_rho_avg_PERT_monotony_max_C2": [dice_rho_avg_PERT_monotony_max_C2],
        
        # "dice_rho_avg_MC_monotony_max_C1": [dice_rho_avg_MC_monotony_max_C1],
        # "dice_rho_avg_MC_monotony_max_C2": [dice_rho_avg_MC_monotony_max_C2],
        # }
        
        # df_temp_n = pd.DataFrame.from_dict(dict_rho_dices_vs_monotonies)
        # df_temp_n.to_csv(general_path_to_save + dataset_name + subset + "_correlations_between_monotonies_of_UM_with_various_GT_and_DICE_OF_SI_AND_GT.csv", index=False)

        # chiude ciclo subset
    #%%
    
"""        
        break
        x = np.arange(0,len(GT_dict_max_values_PERT_C1['th_union_max_dice']),1)
        
        plt.figure(figsize=(17,8))
        plt.suptitle("Class 1, Union Mask: Dice - dataset: " + dataset[:-1] + " , subset: " + subset)
        plt.subplot(121)
        plt.title("Optimal Threshold for dice - MC")
        Y = np.squeeze(np.array(GT_dict_max_values_MC_C1['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(GT_dict_max_values_MC_C1['flag_union_dice'])==True)],
                    np.squeeze(np.array(GT_dict_max_values_MC_C1['th_union_max_dice'])[np.where(np.array(GT_dict_max_values_MC_C1['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.subplot(122)
        plt.title("Optimal Threshold for dice - PERT")
        Y = np.squeeze(np.array(GT_dict_max_values_PERT_C1['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(GT_dict_max_values_PERT_C1['flag_union_dice'])==True)],
                    np.squeeze(np.array(GT_dict_max_values_PERT_C1['th_union_max_dice'])[np.where(np.array(GT_dict_max_values_PERT_C1['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds_dice_C1.png")
        plt.close()
        
        x = np.arange(0,len(GT_dict_max_values_PERT_C2['th_union_max_dice']),1)
        
        plt.figure(figsize=(17,8))
        plt.suptitle("Class 2, Union Mask: Dice - dataset: " + dataset[:-1] + " , subset: " + subset)
        plt.subplot(121)
        plt.title("Optimal Threshold for dice - MC")
        Y = np.squeeze(np.array(GT_dict_max_values_MC_C2['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(GT_dict_max_values_MC_C2['flag_union_dice'])==True)],
                    np.squeeze(np.array(GT_dict_max_values_MC_C2['th_union_max_dice'])[np.where(np.array(GT_dict_max_values_MC_C2['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.subplot(122)
        plt.title("Optimal Threshold for dice - PERT")
        Y = np.squeeze(np.array(GT_dict_max_values_PERT_C2['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(GT_dict_max_values_PERT_C2['flag_union_dice'])==True)],
                    np.squeeze(np.array(GT_dict_max_values_PERT_C2['th_union_max_dice'])[np.where(np.array(GT_dict_max_values_PERT_C2['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds_dice_C2.png")
        plt.close()
        
    break

"""
#%%
"""
    # kernel = np.ones((5, 5), np.uint8) 
    # closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
"""