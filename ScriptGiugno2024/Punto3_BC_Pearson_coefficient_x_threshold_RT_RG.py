# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:38:46 2024

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
# dataset = "Renal PAS glomeruli/"
# subset = "test"
# image = "1004289_35BC_then_mean.png"
N = 20
# radius = 5
c = 3

for dataset in ["Renal PAS glomeruli/","Renal PAS tubuli/"]:
    general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V15/"
    general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
    general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    
    for diff_type in tqdm(["PERT","MC"]):
        if diff_type == "PERT": diff_type_name = "_perturbation"
        if diff_type == "MC": diff_type_name = "_MC"
        for subset in tqdm(["test", "val"]):
            image_list = os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/")
            #%%
            C1_data = np.zeros((len(image_list),11))
            C1_dict_data = {
                    'image_list': [],
                    'SI_dice': []
                    }
            C2_data = np.zeros((len(image_list),11))
            C2_dict_data = {
                    'image_list': [],
                    'SI_dice': []
                    }
            row = 0
            for image in tqdm(os.listdir(general_dataset_path + "DATASET/" + subset + "/" + "manual/")):
                GT_path_2c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
                SI_path_2c = general_results_path_2c  + "/" + subset + "/" + "mask/" + image
                GT_path_3c = general_dataset_path + "DATASET/" + subset + "/" + "manual/" + image
                SI_path_3c = general_results_path_3c  + "/" + subset + "/" + "mask/" + image
                diff_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
                diff_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "mask/" + image + "/"
                softmax_path_2c = general_results_path_2c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
                softmax_path_3c = general_results_path_3c + diff_type_name + "/" + subset + "/" + "softmax/" + image + "/"
                
                #%%
                dataset_name = dataset[:-1] + "_3c/"
                GT_mask = cv2.imread(GT_path_3c, cv2.IMREAD_GRAYSCALE)/255
                GT_mask_C1, GT_mask_C2 = wjb_functions.mask_splitter(GT_mask)
                
                C1_zero = False; C2_zero = False
                if not GT_mask_C1.any(): C1_zero = True
                if not GT_mask_C2.any(): C2_zero = True
                
                if C1_zero == True & C2_zero == True: continue
            
                C1_dict_data['image_list'].append(image)
                C2_dict_data['image_list'].append(image)
                            
                DIM = np.shape(GT_mask)[:2]
                softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_3c, DIM, c, N)
                
                unc_map = wjb_functions.mean_softmax_BC_3(softmax_matrix)
                # unc_map = wjb_functions.mean_BC_map_3(softmax_matrix)
                
                SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
                SI_mask_C1, SI_mask_C2 = wjb_functions.mask_splitter(SI_mask)
                SI_mask_C1 = SI_mask_C1.astype(bool)
                SI_mask_C2 = SI_mask_C2.astype(bool)
                
                
                #%%
                C1_SI_dice = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
                C1_dict_data['SI_dice'].append(C1_SI_dice)
                C2_SI_dice = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
                C2_dict_data['SI_dice'].append(C2_SI_dice)
                
                th_range = np.arange(0,11,1)/10
                column = 0
                for th in th_range:
                    C1_unc_map_th = unc_map*(unc_map>=th)
                    if not 'th_{x}'.format(x=th) in C1_dict_data.keys(): C1_dict_data['th_{x}'.format(x=th)] = []
                    C1_dict_data['th_{x}'.format(x=th)].append(np.nansum(C1_unc_map_th))
                    C1_data[row][column] = np.nansum(C1_unc_map_th)
                    C2_unc_map_th = unc_map*(unc_map>=th)
                    if not 'th_{x}'.format(x=th) in C2_dict_data.keys(): C2_dict_data['th_{x}'.format(x=th)] = []
                    C2_dict_data['th_{x}'.format(x=th)].append(np.nansum(C2_unc_map_th))
                    C2_data[row][column] = np.nansum(C2_unc_map_th)
                    column += 1
                row += 1
            
            if not os.path.isdir(general_path_to_save + dataset_name + diff_type + "/" + subset + "/"): os.makedirs(general_path_to_save + dataset_name + diff_type + "/" + subset + "/")
            
            C1_rho_array = []
            C1_dict_corr_per_th = {}
            C2_rho_array = []
            C2_dict_corr_per_th = {}
            for th in th_range:
                temp = np.array([np.array(C1_dict_data['th_{x}'.format(x=th)]),np.array(C1_dict_data['SI_dice'])]).T
                temp_df = pd.DataFrame(temp)
                C1_rho = temp_df.corr()
                # C1_rho = np.corrcoef(np.array(C1_dict_data['th_{x}'.format(x=th)]),np.array(C1_dict_data['SI_dice']))
                plt.figure(figsize=(10,10))
                plt.scatter(np.array(C1_dict_data['th_{x}'.format(x=th)]),np.array(C1_dict_data['SI_dice']))
                plt.title("(C1) Pearson Coefficient: {x}".format(x=C1_rho[0][1]))
                plt.xlabel("Sum of binary entropy map when greater than threshold {x}".format(x=th))
                plt.ylabel("Dice between Single Inference and Ground Truth")
                if th!=1.0:
                    plt.savefig(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C1_DICExBINMAP_threshold_0{x}".format(x=str(th)[-1]) + "BC_then_mean.png")
                if th==1.0:
                    plt.savefig(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C1_DICExBINMAP_threshold_10" + "BC_then_mean.png")
                plt.close()
                if not 'th_{a}'.format(a=th) in C1_dict_corr_per_th.keys(): C1_dict_corr_per_th['th_{a}'.format(a=th)] = []
                C1_dict_corr_per_th['th_{a}'.format(a=th)].append(C1_rho)
                C1_rho_array.append(C1_rho[0][1])
                
                temp = np.array([np.array(C2_dict_data['th_{x}'.format(x=th)]),np.array(C2_dict_data['SI_dice'])]).T
                temp_df = pd.DataFrame(temp)
                C2_rho = temp_df.corr()
                # C2_rho = np.corrcoef(np.array(C2_dict_data['th_{x}'.format(x=th)]),np.array(C2_dict_data['SI_dice']))
                plt.figure(figsize=(10,10))
                plt.scatter(np.array(C2_dict_data['th_{x}'.format(x=th)]),np.array(C2_dict_data['SI_dice']))
                plt.title("(C2) Pearson Coefficient: {x}".format(x=C2_rho[0][1]))
                plt.xlabel("Sum of binary entropy map when greater than threshold {x}".format(x=th))
                plt.ylabel("Dice between Single Inference and Ground Truth")
                if th!=1.0:
                    plt.savefig(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C2_DICExBINMAP_threshold_0{x}".format(x=str(th)[-1]) + "BC_then_mean.png")
                if th==1.0:
                    plt.savefig(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C2_DICExBINMAP_threshold_10" + "BC_then_mean.png")
                plt.close()
                if not 'th_{a}'.format(a=th) in C2_dict_corr_per_th.keys(): C2_dict_corr_per_th['th_{a}'.format(a=th)] = []
                C2_dict_corr_per_th['th_{a}'.format(a=th)].append(C2_rho)
                C2_rho_array.append(C2_rho[0][1])
            
            C1_rho_array = np.array(C1_rho_array)
            plt.figure(figsize=(10,10))
            plt.plot(th_range,C1_rho_array)
            plt.title("(C1) Rho coefficient x Threshold")
            plt.xlabel("Threshold")
            plt.ylabel("Rho coefficient")
            plt.xlim(0,1)
            plt.savefig(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C1_RHO_x_THRESHOLDBC_then_mean.png")
            plt.close()
            
            C2_rho_array = np.array(C2_rho_array)
            plt.figure(figsize=(10,10))
            plt.plot(th_range,C2_rho_array)
            plt.title("(C2) Rho coefficient x Threshold")
            plt.xlabel("Threshold")
            plt.ylabel("Rho coefficient")
            plt.xlim(0,1)
            plt.savefig(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C2_RHO_x_THRESHOLDBC_then_mean.png")
            plt.close()
            
            pd.DataFrame.from_dict(C1_dict_data).to_csv(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C1_DATAFRAME_sumbinent_x_thresholdBC_then_mean.csv")
            pd.DataFrame.from_dict(C1_dict_corr_per_th).to_csv(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C1_RHO_values_x_thresholdBC_then_mean.csv")
            pd.DataFrame.from_dict(C2_dict_data).to_csv(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C2_DATAFRAME_sumbinent_x_thresholdBC_then_mean.csv")
            pd.DataFrame.from_dict(C2_dict_corr_per_th).to_csv(general_path_to_save + dataset_name + diff_type + "/" + subset + "/" + "C2_RHO_values_x_thresholdBC_then_mean.csv")