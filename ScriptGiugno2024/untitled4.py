# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:00:10 2024

@author: willy
"""

#%% IMPORT

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import measure
import copy
from tqdm import tqdm
import wjb_functions
import pandas as pd

#%% FUNCTIONS
#%%% Metrics
def dice(mask_automatic, mask_manual):
    TP_mask = np.multiply(mask_automatic, mask_manual)
    TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(
        int), TP_mask.astype(int)).astype(bool)
    FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(
        int), TP_mask.astype(int)).astype(bool)
    FN = FN_mask.sum()

    if TP == 0 and FN == 0 and FP == 0:
        dice_ind = np.nan
    else:
        dice_ind = 2*TP/(2*TP+FP+FN)
    return dice_ind

#%%% Mask manipulation

def mask_splitter(mask):
    mask_C1 = np.copy(mask)
    mask_C2 = np.copy(mask)

    mask_C1[mask > 0.8] = 0
    mask_C1[mask_C1 > 0] = 1
    mask_C2[mask < 0.8] = 0
    mask_C2[mask_C2 > 0] = 1
    return mask_C1, mask_C2

#%%% Other

def dilation(mask_3c_int,radius): return cv2.dilate(mask_3c_int,np.ones((radius,radius),np.uint8)).astype(bool)

#%%% MATRIX GENERATOR

def softmax_matrix_gen(softmax_path, DIM, c, N):
    softmax_matrix = np.zeros((DIM[0], DIM[1], c, N), dtype=np.float32)
    counter = 0
    for num in os.listdir(softmax_path):
        for n_class in range(c):
            st0 = np.float32(
                (np.load(softmax_path + "/" + num)['softmax'])[:, :, n_class])
            softmax_matrix[:, :, n_class, counter] = np.copy(st0)
        counter += 1
    return softmax_matrix

#%%% UNCERTAINTY MAPS

def binary_entropy_map(softmax_matrix):
    p_mean = np.mean(softmax_matrix, axis=2)
    p_mean[p_mean == 0] = 1e-8
    p_mean[p_mean == 1] = 1-1e-8
    HB_pert = -(np.multiply(p_mean, np.log2(p_mean)) +
                np.multiply((1-p_mean), np.log2(1-p_mean)))
    HB_pert[np.isnan(HB_pert)] = np.nanmin(HB_pert)
    return HB_pert



#%% Dictionaries

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

#%% PATHS & OPEN IMG + CREAZIONE MASK UNION
path_general = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/"
path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V8/"
radius = 5
# image = '1004289_35.png'
list_of_images_to_look = []


for type_of_diff in type_of_diff_dict:
    list_of_images = os.listdir(path_general + "/k-net+swin/TEST_2classes/RESULTS/test/mask/")
    
    dict_images_th_dices = {
        "image" : [],
        "threshold" : [],
        "dice" : [],
        "baseline_dice" : []
        }

    for image in list_of_images:
        flag = 0
        print("Immagine: " + image)
        SI_path_2c = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_2classes/RESULTS/test/mask/" + image
        SI_path_3c = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_3classes/RESULTS/test/mask/" + image
        GT_path_2c = 'D:/DATASET_Tesi_marzo2024/Liver HE steatosis/DATASET_2classes/test/manual/' + image
        GT_path_3c = 'D:/DATASET_Tesi_marzo2024/Liver HE steatosis/DATASET_3classes/test/manual/' + image
        
        SI_mask_2c = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        SI_mask_3c = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
        SI_mask_3c_ext, SI_mask_3c_int = mask_splitter(SI_mask_3c)
        GT_mask_2c = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
        GT_mask_3c = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
        GT_mask_3c_ext, GT_mask_3c_int = mask_splitter(GT_mask_3c)
        
        if not GT_mask_2c.any(): 
            print("Ground Truth di " + image + " è nullo")
            continue
        
        DIM = np.shape(GT_mask_2c)
        
        path_2c = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_2classes/" + type_of_diff_dict[type_of_diff]["results"] + "/test/mask/" + image + "/"
        path_3c = "D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_3classes/" + type_of_diff_dict[type_of_diff]["results"] + "/test/mask/" + image + "/"
        
        
        #%% CALCOLO VARIE TH
        
        
        path_20_softmax_2c = 'D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_2classes/' + type_of_diff_dict[type_of_diff]["results"] + '/test/softmax/'+ image + "/"
        path_20_softmax_3c = 'D:/DATASET_Tesi_marzo2024/Liver HE steatosis/k-net+swin/TEST_3classes/' + type_of_diff_dict[type_of_diff]["results"] + '/test/softmax/'+ image + "/"
        N = 20
        
        softmax_matrix_2c = softmax_matrix_gen(path_20_softmax_2c, DIM, 2, N)
        softmax_matrix_3c = softmax_matrix_gen(path_20_softmax_3c, DIM, 3, N)
        
        mask_union_int = wjb_functions.mask_union_gen(path_3c)
        mask_avg_int = wjb_functions.mask_avg_gen(softmax_matrix_3c)
        mask_SI_int = SI_mask_3c_int
        
        list_masks_used_to_reduce = ["mask_union","mask_average","single_inference"]
        
        dict_masks = {
            
            "mask_union": {"name": mask_union_int},
            "mask_average": {"name": mask_avg_int},
            "single_inference": {"name": SI_mask_3c_int}
            
            }
        
        # softmax_matrix_PERT_2c[:,:,1,:][GT_mask_3c_ext>0] = 0
        bin_ent_map_2c = binary_entropy_map(softmax_matrix_2c[:,:,1,:])
        BRUM = copy.deepcopy(bin_ent_map_2c)
        
        for mask_to_use in list_masks_used_to_reduce:
            
            MASK_TO_USE_TO_REDUCE = dict_masks[mask_to_use]["name"]
        
            label_image = measure.label(MASK_TO_USE_TO_REDUCE)
            n_objects = label_image.max()
            masks_matrix= np.zeros([label_image.shape[0],label_image.shape[1],n_objects])
            unc_map_matrix = np.copy(masks_matrix)
            tau_array = []
            for i in range(n_objects):
                current_mask = np.copy(label_image)
                current_mask[current_mask!=i+1] = 0
                max_mask = np.max(current_mask)
                if max_mask == 0: max_mask = 1
                current_mask = current_mask/max_mask
                masks_matrix[:,:,i] = current_mask
                unc_map_matrix = np.multiply(current_mask,BRUM)
                tau_i = np.nanmean(unc_map_matrix[unc_map_matrix>0])
                tau_array.append(tau_i)
                
            
            #%% LOOP SU VARIE TH
            
            array = copy.deepcopy(tau_array)
            array = np.array(array)
            th_range = np.array(range(0,21,1))/20 
            
            dice_array_mask_unione=[]
            counter = 0
            masks_of_masks = np.zeros((DIM[0],DIM[1],len(th_range)))
            SI_dice = dice(SI_mask_2c,GT_mask_2c)
            
            for th in th_range:
                array_temp = copy.deepcopy(array)
                array_temp[array>th] = 0
                masks_matrix_temp = masks_matrix[:,:,np.where(array_temp>0)[0]]
                mask_th = np.sum(masks_matrix_temp, axis=-1)
                
                mask_unione_reduced = np.zeros_like(mask_union_int)
                mask_unione_reduced[mask_th>0] = mask_union_int[mask_th>0]
                
                mask_unione_reduced_TO_CALC_DICE = dilation(mask_unione_reduced.astype(np.uint8),radius)
                
                dice_th_mask_unione = dice(mask_unione_reduced_TO_CALC_DICE,GT_mask_2c)
                dice_array_mask_unione.append(dice_th_mask_unione)
                
                if dice_th_mask_unione > SI_dice:
                    if not image in list_of_images_to_look: list_of_images_to_look.append(image)
                    
                    dict_images_th_dices["image"] += [image]
                    dict_images_th_dices["threshold"] += [th]
                    dict_images_th_dices["dice"] += [dice_th_mask_unione]
                    dict_images_th_dices["baseline_dice"] += [SI_dice]
                    
                    flag = 1
                
                
            #%% dice plot
            savepath = path_to_save + "Liver Steatosis HE/" + type_of_diff_dict[type_of_diff]["name"] + "/dilation/" + mask_to_use + "/"
            if not os.path.isdir(savepath): os.makedirs(savepath)
            
            DF = pd.DataFrame.from_dict(dict_images_th_dices)
            DF.to_csv(savepath + "Immagini_salienti_con_threshold_e_dice.csv", index=False)
            
            # plt.figure()
            # plt.title(image)
            # plt.plot(th_range,dice_array_mask_unione, label="Dice x Threshold")
            # plt.axhline(SI_dice, color="r", label="Dice Baseline")
            # plt.xlabel("Threshold")
            # plt.ylabel("Dice")
            # plt.legend()
            # plt.savefig(savepath + image[:-4] + type_of_diff_dict[type_of_diff]["name"] +"_Dice_x_Threshold.png")
            # plt.close()
            
            #%% subplot temp
            
            # plt.figure()
            # plt.subplot(141)
            # plt.imshow(SI_mask_2c)
            # plt.subplot(142)
            # plt.imshow(GT_mask_2c)
            # plt.subplot(143)
            # plt.imshow(mask_union_PR_int)
            # plt.subplot(144)
            # plt.imshow(BRUM)
            
#%% PROVA
index = np.array(dice_array_mask_unione)>0.5
tempx = th_range[index]
tempy = np.asarray(dice_array_mask_unione)[index]
plt.vlines(th_range, 0, dice_array_mask_unione)
plt.scatter(th_range,dice_array_mask_unione)
plt.scatter(tempx,tempy, color='r')
