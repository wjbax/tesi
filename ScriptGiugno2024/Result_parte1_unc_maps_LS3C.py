# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:20:45 2024

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
c = 3

#%%

general_path_to_save = "D:/RESULT_TESI_V2_PARTE1/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

dataset_name = dataset[:-1] + "_3c/"
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
            GT_mask = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
            GT_mask_C1,GT_mask_C2 = wjb_functions.mask_splitter(GT_mask)
            SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
            SI_mask_C1,SI_mask_C2 = wjb_functions.mask_splitter(SI_mask)
            DIM = np.shape(GT_mask)[:2]
            
            C1_SI_dice = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
            C2_SI_dice = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
            
            softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_3c, DIM, c, N)
            
            
            #%% Binary Entropy Map --> BINENT
            C1_BINENT = wjb_functions.binary_entropy_map(softmax_matrix[:,:,1,:])
            C2_BINENT = wjb_functions.binary_entropy_map(softmax_matrix[:,:,2,:])
            
            
            #%% First BC then MEAN --> BC2MEAN
            BC2MEAN = wjb_functions.mean_BC_map_3(softmax_matrix)
            
            # if np.max(BC2MEAN)>1: print("BC2MEAN IMAGE " + image + " SUBSET " + subset + " SUPERA 1")
            
            #%% First softmax mean then BC --> SMEAN2BC
            SMEAN2BC = wjb_functions.mean_softmax_BC_3(softmax_matrix)
            
            # if np.max(SMEAN2BC)>1: print("SMEAN2BC IMAGE " + image + " SUBSET " + subset + " SUPERA 1")
            
            #%%
            path_to_save_figure = general_path_to_save + dataset_name + diff_type + "/" + subset + "/"
            if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
            
            plt.figure(figsize=(24,8))
            plt.suptitle("Dataset: " + dataset[:-1] + ", subset: " + subset + ", image: " + image[:-4])
            plt.subplot(241)
            plt.imshow(OR_image)
            plt.title("Original Image")
            plt.subplot(242)
            plt.imshow(GT_mask)
            plt.title("Ground Truth Mask")
            plt.subplot(243)
            plt.imshow(SI_mask)
            plt.title("Single Inference Mask")
            plt.subplot(245)
            plt.imshow(C1_BINENT)
            plt.title("Binary Entropy Uncertainty Map C1")
            plt.colorbar()
            plt.subplot(246)
            plt.imshow(C2_BINENT)
            plt.title("Binary Entropy Uncertainty Map C2")
            plt.colorbar()
            plt.subplot(247)
            plt.imshow(BC2MEAN)
            plt.title("Mean of BC Maps - Uncertainty Map")
            plt.colorbar()
            plt.subplot(248)
            plt.imshow(SMEAN2BC)
            plt.title("Mean Softmax BC Uncertainty Map")
            plt.colorbar()
            plt.savefig(path_to_save_figure + image)
            plt.close()        
        
