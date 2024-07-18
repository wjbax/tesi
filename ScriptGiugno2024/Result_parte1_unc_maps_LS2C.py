# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:37:17 2024

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

#%%

general_path_to_save = "D:/RESULT_TESI_V2_PARTE1/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

dataset_name = dataset[:-1] + "_2c"
for diff_type in tqdm(["PERT","MC"]):
    if diff_type == "PERT": diff_type_name = "_perturbation"
    if diff_type == "MC": diff_type_name = "_MC"
    for subset in ["test", "val"]:
        image_path = os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")
        for image in tqdm(image_path):
            GT_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
            OR_path_2c = general_dataset_path + "DATASET_2classes/" + subset + "/" + "image/" + image
            SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
            GT_path_3c = general_dataset_path + "DATASET_3classes/" + subset + "/" + "manual/" + image
            SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
            diff_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
            diff_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
            softmax_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
            softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
            
            #%%
            OR_image = cv2.imread(OR_path_2c)
            OR_image = cv2.cvtColor(OR_image, cv2.COLOR_BGR2RGB)
            GT_mask = cv2.imread(GT_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
            SI_mask = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
            SI_dice = wjb_functions.dice(SI_mask,GT_mask)
            DIM = np.shape(GT_mask)[:2]
            
            softmax_matrix_or = wjb_functions.softmax_matrix_gen(softmax_path_2c, DIM, c, N)
            # shape = np.shape(softmax_matrix_or)
            # softmax_matrix_temp = np.zeros((shape[0],shape[1],3,shape[3]))
            # softmax_matrix_temp[:,:,1,:] = copy.deepcopy(softmax_matrix_or[:,:,0,:])
            # softmax_matrix_temp[:,:,2,:] = copy.deepcopy(softmax_matrix_or[:,:,1,:])
            # softmax_matrix = copy.deepcopy(softmax_matrix_temp)
            softmax_matrix = copy.deepcopy(softmax_matrix_or)
            
            
            #%% Binary Entropy Map --> BINENT
            BINENT = wjb_functions.binary_entropy_map(softmax_matrix_or[:,:,1,:])
            
            if np.max(BINENT)>1: print("BINENT IMAGE " + image + " SUBSET " + subset + " SUPERA 1")
            
            #%% First BC then MEAN --> BC2MEAN
            radicando_matrix = np.sqrt(softmax_matrix[:,:,0,:]*softmax_matrix[:,:,1,:])
            radicando_matrix[radicando_matrix<np.sqrt(1e-7)] = np.sqrt(1e-7) 
            bc_map_matrix = np.log(radicando_matrix)/np.log(np.sqrt(1e-7))
            BC2MEAN = np.nanmean(bc_map_matrix,axis=-1)
            
            if np.max(BC2MEAN)>1: print("BC2MEAN IMAGE " + image + " SUBSET " + subset + " SUPERA 1")
            
            #%% First softmax mean then BC --> SMEAN2BC
            mean_softmax = np.nanmean(softmax_matrix_or,axis=-1)
            radicando_map = np.sqrt(mean_softmax[:,:,0]*mean_softmax[:,:,1])
            radicando_map[radicando_map<np.sqrt(1e-7)] = np.sqrt(1e-7)
            SMEAN2BC = np.log(radicando_map)/np.log(np.sqrt(1e-7))
            
            if np.max(SMEAN2BC)>1: print("SMEAN2BC IMAGE " + image + " SUBSET " + subset + " SUPERA 1")
            
            #%%
            path_to_save_figure = general_path_to_save + dataset_name + "/" + diff_type + "/" + subset + "/"
            if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
            
            plt.figure(figsize=(18,8))
            plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4])
            plt.subplot(231)
            plt.imshow(OR_image)
            plt.title("Original Image")
            plt.subplot(232)
            plt.imshow(GT_mask)
            plt.title("Ground Truth Mask")
            plt.subplot(233)
            plt.imshow(SI_mask)
            plt.title("Single Inference Mask")
            plt.subplot(234)
            plt.imshow(BINENT)
            plt.title("Binary Entropy Uncertainty Map")
            plt.colorbar()
            plt.subplot(235)
            plt.imshow(BC2MEAN)
            plt.title("Mean of BC Maps - Uncertainty Map")
            plt.colorbar()
            plt.subplot(236)
            plt.imshow(SMEAN2BC)
            plt.title("Mean Softmax BC Uncertainty Map")
            plt.colorbar()
            plt.savefig(path_to_save_figure + image)
            plt.close()
            
            
            
            
        
        
        
        
        
