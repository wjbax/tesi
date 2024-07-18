# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:52:33 2024

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
# subset = "test"
# image = "1004289_35.png"
N = 20
# radius = 5
c = 2

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V19/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

#%%
# for diff_type in tqdm(["PERT","MC"]):
#     if diff_type == "PERT": diff_type_name = "_perturbation"
#     if diff_type == "MC": diff_type_name = "_MC"
for subset in tqdm(["test", "val"]):
    
    dict_max_values_MC = {
        'image_name': [],
        'union_max_dice': [],
        'th_union_max_dice': [],
        'flag_union_dice': []
        }
    
    dict_max_values_PERT = {
        'image_name': [],
        'union_max_dice': [],
        'th_union_max_dice': [],
        'flag_union_dice': []
        }
    
    for image in tqdm(os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")):
        
        diff_type_name = '_MC'
        diff_type = "MC"
        
        GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
        SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
        GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image
        SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
        diff_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
        diff_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
        softmax_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
        softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
        
        
        #%%
        dataset_name = dataset[:-1] + "_2c/"
        GT_mask = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        if not GT_mask.any(): continue
        
        DIM = np.shape(GT_mask)[:2]
        softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_2c, DIM, c, N)
        unc_map = wjb_functions.binary_entropy_map(softmax_matrix[:,:,1,:])
        
        mask_union = wjb_functions.mask_union_gen(diff_path_2c)
        SI_mask = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        SI_dice = wjb_functions.dice(SI_mask,GT_mask)
        
        th_range = np.arange(1,11,1)/10
        
        union_th_dice_array = []
        
        for th in th_range:
            mask_th_union = mask_union & (unc_map<=th)
            union_th_dice_array.append(wjb_functions.dice(mask_th_union, GT_mask))
            
        union_th_dice_array = np.array(union_th_dice_array)
        
        tabella_temp = np.array([union_th_dice_array]).T
        
        df_temp = pd.DataFrame({'image': image,
                                'union_th_dice_array': tabella_temp[:, 0]
                                })
        
        dict_max_values_MC['image_name'].append(image)
        dict_max_values_MC['union_max_dice'].append(np.nanmax(union_th_dice_array))
        dict_max_values_MC['th_union_max_dice'].append(th_range[np.where(union_th_dice_array == np.nanmax(union_th_dice_array))[0][0]])
        
        if np.max(union_th_dice_array) > SI_dice: 
            dict_max_values_MC['flag_union_dice'].append(1)
        else: 
            dict_max_values_MC['flag_union_dice'].append(0)
        
        
        path_to_save_figure = general_path_to_save + dataset_name + subset + "/"
        if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
        
        df_temp.to_csv(path_to_save_figure + image[:-4] + "_DATAFRAME_" + diff_type +".csv", index = False)
        
        # plt.figure(figsize=(40,8))
        # plt.title("Image " + image + " from dataset " + dataset[:-1] + ", subset: " + subset)
        # plt.plot(th_range, union_th_dice_array, 'b', label="Mask Union")
        # plt.xlim(0,1)
        # plt.ylabel("Dice")
        # plt.xlabel("Threshold")
        # plt.axhline(SI_dice,color='r',linestyle='--',label="BaseLine: DICE SI")
        # plt.legend()
        # plt.savefig(path_to_save_figure + image[:-4] + "_dice_x_threshold_" + diff_type + ".png")
        # plt.close()
        
        union_th_dice_array_MC = np.copy(union_th_dice_array)
        del union_th_dice_array
        
        diff_type_name = '_perturbation'
        diff_type = "PERT"
        
        GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
        SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
        GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image
        SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
        diff_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
        diff_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
        softmax_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
        softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"

        dataset_name = dataset[:-1] + "_2c/"
        GT_mask = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        if not GT_mask.any(): continue
        
        DIM = np.shape(GT_mask)[:2]
        softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_2c, DIM, c, N)
        unc_map = wjb_functions.binary_entropy_map(softmax_matrix[:,:,1,:])
        
        mask_union = wjb_functions.mask_union_gen(diff_path_2c)
        SI_mask = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        
        SI_dice = wjb_functions.dice(SI_mask,GT_mask)
        
        th_range = np.arange(1,11,1)/10
        
        union_th_dice_array = []
        
        for th in th_range:
            mask_th_union = mask_union & (unc_map<=th)
            union_th_dice_array.append(wjb_functions.dice(mask_th_union, GT_mask))
            
        union_th_dice_array = np.array(union_th_dice_array)
        
        tabella_temp = np.array([union_th_dice_array]).T
        
        df_temp = pd.DataFrame({'image': image,
                                'union_th_dice_array': tabella_temp[:, 0]
                                })
        
        dict_max_values_PERT['image_name'].append(image)
        dict_max_values_PERT['union_max_dice'].append(np.nanmax(union_th_dice_array))
        dict_max_values_PERT['th_union_max_dice'].append(th_range[np.where(union_th_dice_array == np.nanmax(union_th_dice_array))[0][0]])
        
        if np.max(union_th_dice_array) > SI_dice: 
            dict_max_values_PERT['flag_union_dice'].append(1)
        else: 
            dict_max_values_PERT['flag_union_dice'].append(0)
        
        
        path_to_save_figure = general_path_to_save + dataset_name + subset + "/"
        if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
        
        df_temp.to_csv(path_to_save_figure + image[:-4] + "_DATAFRAME_" + diff_type +".csv", index = False)
        
        union_th_dice_array_PERT = np.copy(union_th_dice_array)
        del union_th_dice_array
        
        max_MC_PERT = np.max(np.array([union_th_dice_array_PERT,union_th_dice_array_MC]).T,axis=-1)
        
        plt.figure(figsize=(8,8))
        plt.title("Image " + image[:-4] + " from dataset " + dataset[:-1] + ", subset: " + subset)
        plt.plot(th_range, union_th_dice_array_MC, 'b', label="Montecarlo")
        plt.plot(th_range, union_th_dice_array_PERT, 'g', label="Perturbation")
        plt.plot(th_range, max_MC_PERT, 'm', label="Max between MC and PERT")
        plt.axhline(SI_dice,color='r',linestyle='--',label="BaseLine: DICE SI")
        plt.xlim(0,1)
        plt.ylabel("Dice")
        plt.xlabel("Threshold")
        plt.legend()
        plt.savefig(path_to_save_figure + image[:-4] + "_dice_x_threshold.png")
        plt.close()
    
    df_gen = pd.DataFrame.from_dict(dict_max_values_MC)
    df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_MC.csv", index=True)
    del df_gen
    df_gen = pd.DataFrame.from_dict(dict_max_values_PERT)
    df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_PERT.csv", index=True)
    
    x = np.arange(0,len(dict_max_values_PERT['th_union_max_dice']),1)
    
    plt.figure(figsize=(17,8))
    plt.suptitle("Union Mask: Dice - dataset: " + dataset[:-1] + " , subset: " + subset)
    plt.subplot(121)
    plt.title("Optimal Threshold for dice - MC")
    Y = np.squeeze(np.array(dict_max_values_MC['th_union_max_dice']))
    plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
    plt.scatter(x[np.where(np.array(dict_max_values_MC['flag_union_dice'])==True)],
                np.squeeze(np.array(dict_max_values_MC['th_union_max_dice'])[np.where(np.array(dict_max_values_MC['flag_union_dice'])==True)]),color='r',
                label='At this threshold, dice(th) is GREATER than baseline dice')
    plt.xlabel("Image number")
    plt.ylabel("Optimal threshold - dice")
    plt.legend()        
    plt.subplot(122)
    plt.title("Optimal Threshold for dice - PERT")
    Y = np.squeeze(np.array(dict_max_values_PERT['th_union_max_dice']))
    plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
    plt.scatter(x[np.where(np.array(dict_max_values_PERT['flag_union_dice'])==True)],
                np.squeeze(np.array(dict_max_values_PERT['th_union_max_dice'])[np.where(np.array(dict_max_values_PERT['flag_union_dice'])==True)]),color='r',
                label='At this threshold, dice(th) is GREATER than baseline dice')
    plt.xlabel("Image number")
    plt.ylabel("Optimal threshold - dice")
    plt.legend()        
    plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds_dice.png")
    plt.close()

# kernel = np.ones((5, 5), np.uint8) 
# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)