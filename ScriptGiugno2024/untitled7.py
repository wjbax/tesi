# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:01:53 2024

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

diff_type = "PERT"
diff_type_name = "_perturbation/"
dataset = "Liver HE steatosis/"
subset = "test"
image = "1004289_35.png"
N = 20

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V10/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
GT_path = general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/" + image
SI_path = general_results_path  + "/" + subset + "/" + "mask/" + image
diff_path = general_results_path + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
softmax_path = general_results_path + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"

#%%

GT_mask = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE).astype(bool)
SI_mask = cv2.imread(SI_path, cv2.IMREAD_GRAYSCALE).astype(bool)
mask_union = wjb_functions.mask_union_gen(diff_path)
softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path, np.shape(GT_mask)[:2], 2, N)
unc_map = wjb_functions.binary_entropy_map(softmax_matrix[:,:,1,:])

SI_dice = wjb_functions.dice(SI_mask,GT_mask)

th_range = np.arange(0,11,1)/10

for th in th_range:
    mask_th = mask_union & (unc_map<th) # parto da th = 0 -> prendo tutti i valori possibili
    