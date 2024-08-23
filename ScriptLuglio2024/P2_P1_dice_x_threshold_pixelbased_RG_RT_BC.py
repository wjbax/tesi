# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:14:45 2024

@author: willy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:06:57 2024

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
# dataset = "Liver HE steatosis/"
# subset = "test"
# image = "1004289_35.png"
N = 20
# radius = 5
c = 3

#%%
# for diff_type in tqdm(["PERT","MC"]):
#     if diff_type == "PERT": diff_type_name = "_perturbation"
#     if diff_type == "MC": diff_type_name = "_MC"

for dataset in ["Renal PAS glomeruli/", "Renal PAS tubuli/"]:
    
    general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V19_V2BC/"
    general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
    general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    
    for subset in tqdm(["test", "val"]):
        
        dict_max_values_MC_C1 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        dict_max_values_PERT_C1 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        dict_max_values_MC_C2 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        dict_max_values_PERT_C2 = {
            'image_name': [],
            'union_max_dice': [],
            'th_union_max_dice': [],
            'flag_union_dice': []
            }
        
        
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
            
            if not C1_exist and not C2_exist: continue
            
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
            # unc_map_C1_MC = wjb_functions.binary_entropy_map(softmax_matrix_MC[:,:,1,:])
            # unc_map_C2_MC = wjb_functions.binary_entropy_map(softmax_matrix_MC[:,:,2,:])
            softmax_matrix_PERT = wjb_functions.softmax_matrix_gen(softmax_path_3c_PERT, DIM, c, N)
            # unc_map_C1_PERT = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
            # unc_map_C2_PERT = wjb_functions.binary_entropy_map(softmax_matrix_PERT[:,:,2,:])
            unc_map_C1_PERT = wjb_functions.mean_softmax_BC_3(softmax_matrix_PERT)
            unc_map_C1_MC = wjb_functions.mean_softmax_BC_3(softmax_matrix_MC)
            unc_map_C2_PERT = copy.deepcopy(unc_map_C1_PERT)
            unc_map_C2_MC = copy.deepcopy(unc_map_C1_MC)
            
            th_range = np.arange(1,11,1)/10
            
            union_th_dice_array_C1_MC = []
            union_th_dice_array_C2_MC = []
            union_th_dice_array_C1_PERT = []
            union_th_dice_array_C2_PERT = []
            
            # C1           
            
            for th in th_range:
                mask_th_union_C1_MC = mask_union_C1_MC.astype(bool) & (unc_map_C1_MC<=th)
                union_th_dice_array_C1_MC.append(wjb_functions.dice(mask_th_union_C1_MC, GT_mask_C1))
                mask_th_union_C2_MC = mask_union_C2_MC.astype(bool) & (unc_map_C2_MC<=th)
                union_th_dice_array_C2_MC.append(wjb_functions.dice(mask_th_union_C2_MC, GT_mask_C2))
                mask_th_union_C1_PERT = mask_union_C1_PERT.astype(bool) & (unc_map_C1_PERT<=th)
                union_th_dice_array_C1_PERT.append(wjb_functions.dice(mask_th_union_C1_PERT, GT_mask_C1))
                mask_th_union_C2_PERT = mask_union_C2_PERT.astype(bool) & (unc_map_C2_PERT<=th)
                union_th_dice_array_C2_PERT.append(wjb_functions.dice(mask_th_union_C2_PERT, GT_mask_C2))
                
            union_th_dice_array_C1_MC = np.array(union_th_dice_array_C1_MC)
            union_th_dice_array_C2_MC = np.array(union_th_dice_array_C2_MC)
            union_th_dice_array_C1_PERT = np.array(union_th_dice_array_C1_PERT)
            union_th_dice_array_C2_PERT = np.array(union_th_dice_array_C2_PERT)
            
            tabella_temp = np.array([union_th_dice_array_C1_MC,
                                     union_th_dice_array_C2_MC,
                                     union_th_dice_array_C1_PERT,
                                     union_th_dice_array_C2_PERT]).T
            
            df_temp = pd.DataFrame({'image': image,
                                    'union_th_dice_array_C1_MC': tabella_temp[:, 0],
                                    'union_th_dice_array_C2_MC': tabella_temp[:, 1],
                                    'union_th_dice_array_C1_PERT': tabella_temp[:, 2],
                                    'union_th_dice_array_C2_PERT': tabella_temp[:, 3]
                                    })
            
            path_to_save_figure = general_path_to_save + dataset_name + subset + "/"
            if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
            
            if C1_exist:
                dict_max_values_MC_C1['image_name'].append(image)
                dict_max_values_MC_C1['union_max_dice'].append(np.nanmax(union_th_dice_array_C1_MC))
                dict_max_values_MC_C1['th_union_max_dice'].append(th_range[np.where(union_th_dice_array_C1_MC == np.nanmax(union_th_dice_array_C1_MC))[0][0]])
                
                if np.max(union_th_dice_array_C1_MC) > SI_dice_C1: 
                    dict_max_values_MC_C1['flag_union_dice'].append(1)
                else: 
                    dict_max_values_MC_C1['flag_union_dice'].append(0)
                    
                df_gen = pd.DataFrame.from_dict(dict_max_values_MC_C1)
                df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_MC_C1.csv", index=True)
                del df_gen
                
                dict_max_values_PERT_C1['image_name'].append(image)
                dict_max_values_PERT_C1['union_max_dice'].append(np.nanmax(union_th_dice_array_C1_PERT))
                dict_max_values_PERT_C1['th_union_max_dice'].append(th_range[np.where(union_th_dice_array_C1_PERT == np.nanmax(union_th_dice_array_C1_PERT))[0][0]])
                
                if np.max(union_th_dice_array_C1_PERT) > SI_dice_C1: 
                    dict_max_values_PERT_C1['flag_union_dice'].append(1)
                else: 
                    dict_max_values_PERT_C1['flag_union_dice'].append(0)
                    
                df_gen = pd.DataFrame.from_dict(dict_max_values_PERT_C1)
                df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_PERT_C1.csv", index=True)
                del df_gen
                
                title_C1 = "Classe 1"
            else: title_C1 = "Classe 1 - Not present in GT"
                
            if C2_exist:
                
                dict_max_values_MC_C2['image_name'].append(image)
                dict_max_values_MC_C2['union_max_dice'].append(np.nanmax(union_th_dice_array_C2_MC))
                dict_max_values_MC_C2['th_union_max_dice'].append(th_range[np.where(union_th_dice_array_C2_MC == np.nanmax(union_th_dice_array_C2_MC))[0][0]])
                
                if np.max(union_th_dice_array_C2_MC) > SI_dice_C2: 
                    dict_max_values_MC_C2['flag_union_dice'].append(2)
                else: 
                    dict_max_values_MC_C2['flag_union_dice'].append(0)
                    
                df_gen = pd.DataFrame.from_dict(dict_max_values_MC_C2)
                df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_MC_C2.csv", index=True)
                del df_gen
                    
                dict_max_values_PERT_C2['image_name'].append(image)
                dict_max_values_PERT_C2['union_max_dice'].append(np.nanmax(union_th_dice_array_C2_PERT))
                dict_max_values_PERT_C2['th_union_max_dice'].append(th_range[np.where(union_th_dice_array_C2_PERT == np.nanmax(union_th_dice_array_C2_PERT))[0][0]])
                
                if np.max(union_th_dice_array_C2_PERT) > SI_dice_C2: 
                    dict_max_values_PERT_C2['flag_union_dice'].append(2)
                else: 
                    dict_max_values_PERT_C2['flag_union_dice'].append(0)
                
                df_gen = pd.DataFrame.from_dict(dict_max_values_PERT_C2)
                df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_PERT_C2.csv", index=True)
                del df_gen
                title_C2 = "Classe 2"
            else: title_C2 = "Classe 2 - Not present in GT"
            
            plt.figure(figsize=(16,8))
            plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4])
            plt.subplot(121)
            plt.title(title_C1)
            plt.plot(th_range,union_th_dice_array_C1_MC, 'g', label="Montecarlo")
            plt.plot(th_range,union_th_dice_array_C1_PERT, 'b', label="Perturbation")
            plt.axhline(SI_dice_C1, label="Baseline SI vs GT", color='r', linestyle='--')
            plt.legend(loc=4)
            plt.subplot(122)
            plt.title(title_C2)
            plt.plot(th_range,union_th_dice_array_C2_MC, 'g', label="Montecarlo")
            plt.plot(th_range,union_th_dice_array_C2_PERT, 'b', label="Perturbation")
            plt.axhline(SI_dice_C2, label="Baseline SI vs GT", color='r', linestyle='--')
            plt.legend(loc=4)
            plt.savefig(path_to_save_figure + image[:-4] + "_dice_x_threshold_MC_or_PERT.png")
            plt.close()
            
            max_C1 = np.max(np.array([union_th_dice_array_C1_MC,union_th_dice_array_C1_PERT]).T,axis=-1)
            max_C2 = np.max(np.array([union_th_dice_array_C2_MC,union_th_dice_array_C2_PERT]).T,axis=-1)

            plt.figure(figsize=(16,8))
            plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4])
            plt.subplot(121)
            plt.title(title_C1)
            plt.plot(th_range,max_C1, 'b', label="Max between MC and PERT")
            plt.axhline(SI_dice_C1, label="Baseline SI vs GT", color='r', linestyle='--')
            plt.legend(loc=4)
            plt.subplot(122)
            plt.title(title_C2)
            plt.plot(th_range,max_C2, 'b', label="Max between MC and PERT")
            plt.axhline(SI_dice_C2, label="Baseline SI vs GT", color='r', linestyle='--')
            plt.legend(loc=4)
            plt.savefig(path_to_save_figure + image[:-4] + "_dice_x_threshold_max_MC_PERT.png")
            plt.close()
            
            
        x = np.arange(0,len(dict_max_values_PERT_C1['th_union_max_dice']),1)
        
        plt.figure(figsize=(17,8))
        plt.suptitle("Class 1, Union Mask: Dice - dataset: " + dataset[:-1] + " , subset: " + subset)
        plt.subplot(121)
        plt.title("Optimal Threshold for dice - MC")
        Y = np.squeeze(np.array(dict_max_values_MC_C1['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(dict_max_values_MC_C1['flag_union_dice'])==True)],
                    np.squeeze(np.array(dict_max_values_MC_C1['th_union_max_dice'])[np.where(np.array(dict_max_values_MC_C1['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.subplot(122)
        plt.title("Optimal Threshold for dice - PERT")
        Y = np.squeeze(np.array(dict_max_values_PERT_C1['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(dict_max_values_PERT_C1['flag_union_dice'])==True)],
                    np.squeeze(np.array(dict_max_values_PERT_C1['th_union_max_dice'])[np.where(np.array(dict_max_values_PERT_C1['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds_dice_C1.png")
        plt.close()
        
        x = np.arange(0,len(dict_max_values_PERT_C2['th_union_max_dice']),1)
        
        plt.figure(figsize=(17,8))
        plt.suptitle("Class 2, Union Mask: Dice - dataset: " + dataset[:-1] + " , subset: " + subset)
        plt.subplot(121)
        plt.title("Optimal Threshold for dice - MC")
        Y = np.squeeze(np.array(dict_max_values_MC_C2['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(dict_max_values_MC_C2['flag_union_dice'])==True)],
                    np.squeeze(np.array(dict_max_values_MC_C2['th_union_max_dice'])[np.where(np.array(dict_max_values_MC_C2['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.subplot(122)
        plt.title("Optimal Threshold for dice - PERT")
        Y = np.squeeze(np.array(dict_max_values_PERT_C2['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(dict_max_values_PERT_C2['flag_union_dice'])==True)],
                    np.squeeze(np.array(dict_max_values_PERT_C2['th_union_max_dice'])[np.where(np.array(dict_max_values_PERT_C2['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Optimal threshold - dice")
        plt.legend()        
        plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds_dice_C2.png")
        plt.close()
#%%
"""
    # kernel = np.ones((5, 5), np.uint8) 
    # closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
"""