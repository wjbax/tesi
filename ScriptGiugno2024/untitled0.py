# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:44:30 2024

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
# radius = 5
c = 2

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V11/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

image_list = os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")
#%%
data = np.zeros((len(image_list),10))
dict_data = {
        'image_list': [],
        'SI_dice': []
        }
row = 0
for image in tqdm(os.listdir(general_dataset_path + "DATASET_2classes/" + subset + "/" + "manual/")):
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
    dict_data['image_list'].append(image)
    
    SI_mask = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
    DIM = np.shape(GT_mask)[:2]
    softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_2c, DIM, c, N)
    unc_map = wjb_functions.binary_entropy_map(softmax_matrix[:,:,1,:])
    
    #%%
    SI_dice = wjb_functions.dice(SI_mask,GT_mask)
    dict_data['SI_dice'].append(SI_dice)
    
    th_range = np.arange(0,10,1)/10
    column = 0
    for th in th_range:
        unc_map_th = unc_map*np.where(unc_map>th,1,0)
        if not 'th_{x}'.format(x=th) in dict_data.keys(): dict_data['th_{x}'.format(x=th)] = []
        dict_data['th_{x}'.format(x=th)].append(np.sum(unc_map_th))
        data[row][column] = np.sum(unc_map_th)
        column += 1
    row += 1

rho_array = []
dict_corr_per_th = {}
for th in th_range:
    rho = np.corrcoef(np.array(dict_data['th_{x}'.format(x=th)]),np.array(dict_data['SI_dice']))
    if not 'th_{a}'.format(a=th) in dict_corr_per_th.keys(): dict_corr_per_th['th_{a}'.format(a=th)] = []
    dict_corr_per_th['th_{a}'.format(a=th)].append(rho)
    rho_array.append(rho[0][1])
    
rho_array = np.array(rho_array)
plt.plot(th_range,rho_array)
plt.title("Pearson Coefficient depending on Threshold")