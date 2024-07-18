# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:39:56 2024

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
# dataset = "Renal PAS glomeruli/"
# subset = "test"
# image = "1004289_35.png"
N = 20
# radius = 5
c = 3

#%%
for dataset in ["Renal PAS tubuli/", "Renal PAS glomeruli/"]:
    general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V13/"
    general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
    general_results_path_2c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    general_results_path_3c = general_dataset_path + "k-net+swin/TEST/RESULTS"
    for diff_type in tqdm(["PERT","MC"]):
        if diff_type == "PERT": diff_type_name = "_perturbation"
        if diff_type == "MC": diff_type_name = "_MC"
        for subset in tqdm(["test", "val"]):
            C1_dict_max_values = {
                'image_name': [],
                'union_max_dice': [],
                'th_union_max_dice': [],
                'union_max_recall': [],
                'th_union_max_recall': [],
                'union_max_precision': [],
                'th_union_max_precision': [],
                'avg_max_dice': [],
                'th_avg_max_dice': [],
                'avg_max_recall': [],
                'th_avg_max_recall': [],
                'avg_max_precision': [],
                'th_avg_max_precision': [],
                'SI_max_dice': [],
                'th_SI_max_dice': [],
                'SI_max_recall': [],
                'th_SI_max_recall': [],
                'SI_max_precision': [],
                'th_SI_max_precision': [],
                'flag_union_dice': [],
                'flag_union_recall': [],
                'flag_union_precision': [],
                'flag_avg_dice': [],
                'flag_avg_recall': [],
                'flag_avg_precision': [],
                'flag_SI_dice': [],
                'flag_SI_recall': [],
                'flag_SI_precision': []
                }
            C2_dict_max_values = {
                'image_name': [],
                'union_max_dice': [],
                'th_union_max_dice': [],
                'union_max_recall': [],
                'th_union_max_recall': [],
                'union_max_precision': [],
                'th_union_max_precision': [],
                'avg_max_dice': [],
                'th_avg_max_dice': [],
                'avg_max_recall': [],
                'th_avg_max_recall': [],
                'avg_max_precision': [],
                'th_avg_max_precision': [],
                'SI_max_dice': [],
                'th_SI_max_dice': [],
                'SI_max_recall': [],
                'th_SI_max_recall': [],
                'SI_max_precision': [],
                'th_SI_max_precision': [],
                'flag_union_dice': [],
                'flag_union_recall': [],
                'flag_union_precision': [],
                'flag_avg_dice': [],
                'flag_avg_recall': [],
                'flag_avg_precision': [],
                'flag_SI_dice': [],
                'flag_SI_recall': [],
                'flag_SI_precision': []
                }
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
                            
                DIM = np.shape(GT_mask)[:2]
                softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_3c, DIM, c, N)
                
                
                unc_map = wjb_functions.mean_softmax_BC_3(softmax_matrix)
                # unc_map = wjb_functions.mean_BC_map_3(softmax_matrix)
                
                mask_union = wjb_functions.mask_union_gen_3c(diff_path_3c)/255
                mask_union_C1, mask_union_C2 = wjb_functions.mask_splitter(mask_union)
                mask_union_C1 = mask_union_C1.astype(bool)
                mask_union_C2 = mask_union_C2.astype(bool)
                
                mask_avg = wjb_functions.mask_avg_gen_3c(softmax_matrix)/2
                mask_avg_C1, mask_avg_C2 = wjb_functions.mask_splitter(mask_avg)
                mask_avg_C1 = mask_avg_C1.astype(bool)
                mask_avg_C2 = mask_avg_C2.astype(bool)
                
                SI_mask = cv2.imread(SI_path_3c, cv2.IMREAD_GRAYSCALE)/255
                SI_mask_C1, SI_mask_C2 = wjb_functions.mask_splitter(SI_mask)
                SI_mask_C1 = SI_mask_C1.astype(bool)
                SI_mask_C2 = SI_mask_C2.astype(bool)
                
                SI_dice_C1 = wjb_functions.dice(SI_mask_C1,GT_mask_C1)
                SI_dice_C2 = wjb_functions.dice(SI_mask_C2,GT_mask_C2)
    
                SI_recall_C1 = wjb_functions.recall(SI_mask_C1,GT_mask_C1)
                SI_recall_C2 = wjb_functions.recall(SI_mask_C2,GT_mask_C2)
    
                SI_precision_C1 = wjb_functions.precision(SI_mask_C1,GT_mask_C1)
                SI_precision_C2 = wjb_functions.precision(SI_mask_C2,GT_mask_C2)
    
                th_range = np.arange(1,11,1)/10
                
                C1_union_th_dice_array = []
                C1_avg_th_dice_array = []
                C1_SI_th_dice_array = []
                
                C1_union_th_recall_array = []
                C1_avg_th_recall_array = []
                C1_SI_th_recall_array = []
                
                C1_union_th_precision_array = []
                C1_avg_th_precision_array = []
                C1_SI_th_precision_array = []
                
                C2_union_th_dice_array = []
                C2_avg_th_dice_array = []
                C2_SI_th_dice_array = []
                
                C2_union_th_recall_array = []
                C2_avg_th_recall_array = []
                C2_SI_th_recall_array = []
                
                C2_union_th_precision_array = []
                C2_avg_th_precision_array = []
                C2_SI_th_precision_array = []
                
                for th in th_range:
                    
                    mask_th_union = mask_union * (unc_map<=th)
                    mask_th_avg = mask_avg * (unc_map<=th)
                    mask_th_SI = SI_mask * (unc_map<=th)
                    
                    C1_mask_th_union, C2_mask_th_union = wjb_functions.mask_splitter(mask_th_union)
                    C1_mask_th_avg, C2_mask_th_avg = wjb_functions.mask_splitter(mask_th_avg)
                    C1_mask_th_SI, C2_mask_th_SI = wjb_functions.mask_splitter(mask_th_SI)
                    
                    C1_union_th_dice_array.append(wjb_functions.dice(C1_mask_th_union, GT_mask_C1))
                    C1_avg_th_dice_array.append(wjb_functions.dice(C1_mask_th_avg, GT_mask_C1))
                    C1_SI_th_dice_array.append(wjb_functions.dice(C1_mask_th_SI, GT_mask_C1))
                    
                    C1_union_th_recall_array.append(wjb_functions.recall(C1_mask_th_union,GT_mask_C1))
                    C1_avg_th_recall_array.append(wjb_functions.recall(C1_mask_th_avg,GT_mask_C1))
                    C1_SI_th_recall_array.append(wjb_functions.recall(C1_mask_th_SI,GT_mask_C1))
                    
                    C1_union_th_precision_array.append(wjb_functions.precision(C1_mask_th_union,GT_mask_C1))
                    C1_avg_th_precision_array.append(wjb_functions.precision(C1_mask_th_avg,GT_mask_C1))
                    C1_SI_th_precision_array.append(wjb_functions.precision(C1_mask_th_SI,GT_mask_C1))
                    
                    
                    C2_union_th_dice_array.append(wjb_functions.dice(C2_mask_th_union, GT_mask_C2))
                    C2_avg_th_dice_array.append(wjb_functions.dice(C2_mask_th_avg, GT_mask_C2))
                    C2_SI_th_dice_array.append(wjb_functions.dice(C2_mask_th_SI, GT_mask_C2))
                    
                    C2_union_th_recall_array.append(wjb_functions.recall(C2_mask_th_union,GT_mask_C2))
                    C2_avg_th_recall_array.append(wjb_functions.recall(C2_mask_th_avg,GT_mask_C2))
                    C2_SI_th_recall_array.append(wjb_functions.recall(C2_mask_th_SI,GT_mask_C2))
                    
                    C2_union_th_precision_array.append(wjb_functions.precision(C2_mask_th_union,GT_mask_C2))
                    C2_avg_th_precision_array.append(wjb_functions.precision(C2_mask_th_avg,GT_mask_C2))
                    C2_SI_th_precision_array.append(wjb_functions.precision(C2_mask_th_SI,GT_mask_C2))
                    
                    
                C1_union_th_dice_array = np.array(C1_union_th_dice_array)
                C1_avg_th_dice_array = np.array(C1_avg_th_dice_array)
                C1_SI_th_dice_array = np.array(C1_SI_th_dice_array)
                
                C1_union_th_recall_array = np.array(C1_union_th_recall_array)
                C1_avg_th_recall_array = np.array(C1_avg_th_recall_array)
                C1_SI_th_recall_array = np.array(C1_SI_th_recall_array)
                
                C1_union_th_precision_array = np.array(C1_union_th_precision_array)
                C1_avg_th_precision_array = np.array(C1_avg_th_precision_array)
                C1_SI_th_precision_array = np.array(C1_SI_th_precision_array)
                
                C2_union_th_dice_array = np.array(C2_union_th_dice_array)
                C2_avg_th_dice_array = np.array(C2_avg_th_dice_array)
                C2_SI_th_dice_array = np.array(C2_SI_th_dice_array)
                
                C2_union_th_recall_array = np.array(C2_union_th_recall_array)
                C2_avg_th_recall_array = np.array(C2_avg_th_recall_array)
                C2_SI_th_recall_array = np.array(C2_SI_th_recall_array)
                
                C2_union_th_precision_array = np.array(C2_union_th_precision_array)
                C2_avg_th_precision_array = np.array(C2_avg_th_precision_array)
                C2_SI_th_precision_array = np.array(C2_SI_th_precision_array)
                
                C1_tabella_temp = np.array([C1_union_th_dice_array,
                                         C1_union_th_recall_array,
                                         C1_union_th_precision_array,
                                         C1_avg_th_dice_array,
                                         C1_avg_th_recall_array,
                                         C1_avg_th_precision_array,
                                         C1_SI_th_dice_array,
                                         C1_SI_th_recall_array,
                                         C1_SI_th_precision_array
                                         ]).T
                
                C2_tabella_temp = np.array([C2_union_th_dice_array,
                                         C2_union_th_recall_array,
                                         C2_union_th_precision_array,
                                         C2_avg_th_dice_array,
                                         C2_avg_th_recall_array,
                                         C2_avg_th_precision_array,
                                         C2_SI_th_dice_array,
                                         C2_SI_th_recall_array,
                                         C2_SI_th_precision_array
                                         ]).T
                
                C1_df_temp = pd.DataFrame({'image': image,
                                        'union_th_dice_array': C1_tabella_temp[:, 0], 
                                        'union_th_recall_array': C1_tabella_temp[:, 1],
                                        'union_th_precision_array': C1_tabella_temp[:, 2],
                                        'avg_th_dice_array': C1_tabella_temp[:, 3],
                                        'avg_th_recall_array': C1_tabella_temp[:, 4],
                                        'avg_th_precision_array': C1_tabella_temp[:, 5],
                                        'SI_th_dice_array': C1_tabella_temp[:, 6],
                                        'SI_th_recall_array': C1_tabella_temp[:, 7],
                                        'SI_th_precision_array': C1_tabella_temp[:, 8]
                                        })
                
                C2_df_temp = pd.DataFrame({'image': image,
                                        'union_th_dice_array': C2_tabella_temp[:, 0], 
                                        'union_th_recall_array': C2_tabella_temp[:, 1],
                                        'union_th_precision_array': C2_tabella_temp[:, 2],
                                        'avg_th_dice_array': C2_tabella_temp[:, 3],
                                        'avg_th_recall_array': C2_tabella_temp[:, 4],
                                        'avg_th_precision_array': C2_tabella_temp[:, 5],
                                        'SI_th_dice_array': C2_tabella_temp[:, 6],
                                        'SI_th_recall_array': C2_tabella_temp[:, 7],
                                        'SI_th_precision_array': C2_tabella_temp[:, 8]
                                        })
                
                C1_dict_max_values['image_name'].append(image)
                
                C1_dict_max_values['union_max_dice'].append(np.nanmax(C1_union_th_dice_array))
                if not C1_union_th_dice_array.all()==True:
                    C1_dict_max_values['th_union_max_dice'].append(th_range[np.where(C1_union_th_dice_array == np.nanmax(C1_union_th_dice_array))[0][0]])
                else: C1_dict_max_values['th_union_max_dice'].append(0)
                C1_dict_max_values['union_max_recall'].append(np.nanmax(C1_union_th_recall_array))
                if not C1_union_th_recall_array.all()==True:
                    C1_dict_max_values['th_union_max_recall'].append(th_range[np.where(C1_union_th_recall_array == np.nanmax(C1_union_th_recall_array))[0][0]])
                else: C1_dict_max_values['th_union_max_recall'].append(0)
                C1_dict_max_values['union_max_precision'].append(np.nanmax(C1_union_th_precision_array))
                if not C1_union_th_precision_array.all()== True:
                    C1_dict_max_values['th_union_max_precision'].append(th_range[np.where(C1_union_th_precision_array == np.nanmax(C1_union_th_precision_array))[0][-1]])
                else: C1_dict_max_values['th_union_max_precision'].append(1)
                
                C1_dict_max_values['avg_max_dice'].append(np.nanmax(C1_avg_th_dice_array))
                if not C1_avg_th_dice_array.all()==True:
                    C1_dict_max_values['th_avg_max_dice'].append(th_range[np.where(C1_avg_th_dice_array == np.nanmax(C1_avg_th_dice_array))[0][0]])
                else: C1_dict_max_values['th_avg_max_dice'].append(0)
                C1_dict_max_values['avg_max_recall'].append(np.nanmax(C1_avg_th_recall_array))
                if not C1_avg_th_recall_array.all()==True:
                    C1_dict_max_values['th_avg_max_recall'].append(th_range[np.where(C1_avg_th_recall_array == np.nanmax(C1_avg_th_recall_array))[0][0]])
                else: C1_dict_max_values['th_avg_max_recall'].append(0)
                C1_dict_max_values['avg_max_precision'].append(np.nanmax(C1_avg_th_precision_array))
                if not C1_avg_th_precision_array.all() == True:
                    C1_dict_max_values['th_avg_max_precision'].append(th_range[np.where(C1_avg_th_precision_array == np.nanmax(C1_avg_th_precision_array))[0][-1]])
                else: C1_dict_max_values['th_avg_max_precision'].append(1)
                
                C1_dict_max_values['SI_max_dice'].append(np.nanmax(C1_SI_th_dice_array))
                if not C1_SI_th_dice_array.all()==True:
                    C1_dict_max_values['th_SI_max_dice'].append(th_range[np.where(C1_SI_th_dice_array == np.nanmax(C1_SI_th_dice_array))[0][0]])
                else: C1_dict_max_values['th_SI_max_dice'].append(0)
                C1_dict_max_values['SI_max_recall'].append(np.nanmax(C1_SI_th_recall_array))
                if not C1_SI_th_recall_array.all()==True:
                    C1_dict_max_values['th_SI_max_recall'].append(th_range[np.where(C1_SI_th_recall_array == np.nanmax(C1_SI_th_recall_array))[0][0]])
                else: C1_dict_max_values['th_SI_max_recall'].append(0)
                C1_dict_max_values['SI_max_precision'].append(np.nanmax(C1_SI_th_precision_array))
                if not C1_avg_th_precision_array.all() == True:
                    C1_dict_max_values['th_SI_max_precision'].append(th_range[np.where(C1_SI_th_precision_array == np.nanmax(C1_SI_th_precision_array))[0][-1]])
                else: C1_dict_max_values['th_SI_max_precision'].append(1)
                
                
                if np.max(C1_union_th_dice_array) > SI_dice_C1: 
                    C1_dict_max_values['flag_union_dice'].append(1)
                else: 
                    C1_dict_max_values['flag_union_dice'].append(0)
                
                if np.max(C1_union_th_recall_array) > SI_recall_C1: 
                    C1_dict_max_values['flag_union_recall'].append(1)
                else: 
                    C1_dict_max_values['flag_union_recall'].append(0)
                
                if np.max(C1_union_th_precision_array) > SI_precision_C1: 
                    C1_dict_max_values['flag_union_precision'].append(1)
                else: 
                    C1_dict_max_values['flag_union_precision'].append(0)
                    
                if np.max(C1_avg_th_dice_array) > SI_dice_C1: 
                    C1_dict_max_values['flag_avg_dice'].append(1)
                else: 
                    C1_dict_max_values['flag_avg_dice'].append(0)
                
                if np.max(C1_avg_th_recall_array) > SI_recall_C1: 
                    C1_dict_max_values['flag_avg_recall'].append(1)
                else: 
                    C1_dict_max_values['flag_avg_recall'].append(0)
                
                if np.max(C1_avg_th_precision_array) > SI_precision_C1: 
                    C1_dict_max_values['flag_avg_precision'].append(1)
                else: 
                    C1_dict_max_values['flag_avg_precision'].append(0)
                    
                if np.max(C1_SI_th_dice_array) > SI_dice_C1: 
                    C1_dict_max_values['flag_SI_dice'].append(1)
                else: 
                    C1_dict_max_values['flag_SI_dice'].append(0)
                
                if np.max(C1_SI_th_recall_array) > SI_recall_C1: 
                    C1_dict_max_values['flag_SI_recall'].append(1)
                else: 
                    C1_dict_max_values['flag_SI_recall'].append(0)
                
                if np.max(C1_SI_th_precision_array) > SI_precision_C1: 
                    C1_dict_max_values['flag_SI_precision'].append(1)
                else: 
                    C1_dict_max_values['flag_SI_precision'].append(0)
                
                
                C2_dict_max_values['image_name'].append(image)
                
                C2_dict_max_values['union_max_dice'].append(np.nanmax(C2_union_th_dice_array))
                if not C2_union_th_dice_array.all()==True:
                    C2_dict_max_values['th_union_max_dice'].append(th_range[np.where(C2_union_th_dice_array == np.nanmax(C2_union_th_dice_array))[0][0]])
                else: C2_dict_max_values['th_union_max_dice'].append(0)
                C2_dict_max_values['union_max_recall'].append(np.nanmax(C2_union_th_recall_array))
                if not C2_union_th_recall_array.all()==True:
                    C2_dict_max_values['th_union_max_recall'].append(th_range[np.where(C2_union_th_recall_array == np.nanmax(C2_union_th_recall_array))[0][0]])
                else: C2_dict_max_values['th_union_max_recall'].append(0)
                C2_dict_max_values['union_max_precision'].append(np.nanmax(C2_union_th_precision_array))
                if not C2_union_th_precision_array.all()== True:
                    C2_dict_max_values['th_union_max_precision'].append(th_range[np.where(C2_union_th_precision_array == np.nanmax(C2_union_th_precision_array))[0][-1]])
                else: C2_dict_max_values['th_union_max_precision'].append(1)
                
                C2_dict_max_values['avg_max_dice'].append(np.nanmax(C2_avg_th_dice_array))
                if not C2_avg_th_dice_array.all()==True:
                    C2_dict_max_values['th_avg_max_dice'].append(th_range[np.where(C2_avg_th_dice_array == np.nanmax(C2_avg_th_dice_array))[0][0]])
                else: C2_dict_max_values['th_avg_max_dice'].append(0)
                C2_dict_max_values['avg_max_recall'].append(np.nanmax(C2_avg_th_recall_array))
                if not C2_avg_th_recall_array.all()==True:
                    C2_dict_max_values['th_avg_max_recall'].append(th_range[np.where(C2_avg_th_recall_array == np.nanmax(C2_avg_th_recall_array))[0][0]])
                else: C2_dict_max_values['th_avg_max_recall'].append(0)
                C2_dict_max_values['avg_max_precision'].append(np.nanmax(C2_avg_th_precision_array))
                if not C2_avg_th_precision_array.all() == True:
                    C2_dict_max_values['th_avg_max_precision'].append(th_range[np.where(C2_avg_th_precision_array == np.nanmax(C2_avg_th_precision_array))[0][-1]])
                else: C2_dict_max_values['th_avg_max_precision'].append(1)
                
                C2_dict_max_values['SI_max_dice'].append(np.nanmax(C2_SI_th_dice_array))
                if not C2_SI_th_dice_array.all()==True:
                    C2_dict_max_values['th_SI_max_dice'].append(th_range[np.where(C2_SI_th_dice_array == np.nanmax(C2_SI_th_dice_array))[0][0]])
                else: C2_dict_max_values['th_SI_max_dice'].append(0)
                C2_dict_max_values['SI_max_recall'].append(np.nanmax(C2_SI_th_recall_array))
                if not C2_SI_th_recall_array.all()==True:
                    C2_dict_max_values['th_SI_max_recall'].append(th_range[np.where(C2_SI_th_recall_array == np.nanmax(C2_SI_th_recall_array))[0][0]])
                else: C2_dict_max_values['th_SI_max_recall'].append(0)
                C2_dict_max_values['SI_max_precision'].append(np.nanmax(C2_SI_th_precision_array))
                if not C2_SI_th_precision_array.all() == True:
                    C2_dict_max_values['th_SI_max_precision'].append(th_range[np.where(C2_SI_th_precision_array == np.nanmax(C2_SI_th_precision_array))[0][-1]])
                else: C2_dict_max_values['th_SI_max_precision'].append(1)
                
                
                if np.max(C2_union_th_dice_array) > SI_dice_C2: 
                    C2_dict_max_values['flag_union_dice'].append(1)
                else: 
                    C2_dict_max_values['flag_union_dice'].append(0)
                
                if np.max(C2_union_th_recall_array) > SI_recall_C2: 
                    C2_dict_max_values['flag_union_recall'].append(1)
                else: 
                    C2_dict_max_values['flag_union_recall'].append(0)
                
                if np.max(C2_union_th_precision_array) > SI_precision_C2: 
                    C2_dict_max_values['flag_union_precision'].append(1)
                else: 
                    C2_dict_max_values['flag_union_precision'].append(0)
                    
                if np.max(C2_avg_th_dice_array) > SI_dice_C2: 
                    C2_dict_max_values['flag_avg_dice'].append(1)
                else: 
                    C2_dict_max_values['flag_avg_dice'].append(0)
                
                if np.max(C2_avg_th_recall_array) > SI_recall_C2: 
                    C2_dict_max_values['flag_avg_recall'].append(1)
                else: 
                    C2_dict_max_values['flag_avg_recall'].append(0)
                
                if np.max(C2_avg_th_precision_array) > SI_precision_C2: 
                    C2_dict_max_values['flag_avg_precision'].append(1)
                else: 
                    C2_dict_max_values['flag_avg_precision'].append(0)
                    
                if np.max(C2_SI_th_dice_array) > SI_dice_C2: 
                    C2_dict_max_values['flag_SI_dice'].append(1)
                else: 
                    C2_dict_max_values['flag_SI_dice'].append(0)
                
                if np.max(C2_SI_th_recall_array) > SI_recall_C2: 
                    C2_dict_max_values['flag_SI_recall'].append(1)
                else: 
                    C2_dict_max_values['flag_SI_recall'].append(0)
                
                if np.max(C2_SI_th_precision_array) > SI_precision_C2: 
                    C2_dict_max_values['flag_SI_precision'].append(1)
                else: 
                    C2_dict_max_values['flag_SI_precision'].append(0)
                
                
                path_to_save_figure = general_path_to_save + dataset_name + diff_type + "/" + subset + "/"
                if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
                
                C1_df_temp.to_csv(path_to_save_figure + image[:-4] + "_DATAFRAME_C1_mean_then_BC.csv", index = False)
                
                plt.figure(figsize=(30,8))
                plt.suptitle("Image " + image + " from dataset " + dataset[:-1] + ", subset: " + subset + ", class: C1")
                plt.subplot(131)
                plt.title("dice x threshold")
                plt.plot(th_range, C1_union_th_dice_array, 'b', label="Mask Union")
                plt.plot(th_range, C1_avg_th_dice_array, 'g', label="Mask Avg")
                plt.plot(th_range, C1_SI_th_dice_array, 'm', label="Mask SI")
                plt.xlim(0,1)
                plt.ylabel("Dice")
                plt.xlabel("Threshold")
                plt.axhline(SI_dice_C1,color='r',linestyle='--',label="BaseLine: DICE SI")
                plt.legend()
                plt.subplot(132)
                plt.title("recall x threshold")
                plt.plot(th_range, C1_union_th_recall_array, 'b', label="Mask Union")
                plt.plot(th_range, C1_avg_th_recall_array, 'g', label="Mask Avg")
                plt.plot(th_range, C1_SI_th_recall_array, 'm', label="Mask SI")
                plt.ylabel("Recall")
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.axhline(SI_recall_C1,color='r',linestyle='--',label="BaseLine: recall SI")
                plt.legend()
                plt.subplot(133)
                plt.title("precision x threshold")
                plt.plot(th_range, C1_union_th_precision_array, 'b', label="Mask Union")
                plt.plot(th_range, C1_avg_th_precision_array, 'g', label="Mask Avg")
                plt.plot(th_range, C1_SI_th_precision_array, 'm', label="Mask SI")
                plt.ylabel("Precision")
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.axhline(SI_precision_C1,color='r',linestyle='--',label="BaseLine: precision SI")
                plt.legend()
                plt.savefig(path_to_save_figure + image[:-4] + "_dicerecallprecision_x_threshold_C1_mean_then_BC.png")
                plt.close()
                
                C2_df_temp.to_csv(path_to_save_figure + image[:-4] + "_DATAFRAME_C2_mean_then_BC.csv", index = False)
                
                plt.figure(figsize=(30,8))
                plt.suptitle("Image " + image + " from dataset " + dataset[:-1] + ", subset: " + subset + ", class: C2")
                plt.subplot(131)
                plt.title("dice x threshold")
                plt.plot(th_range, C2_union_th_dice_array, 'b', label="Mask Union")
                plt.plot(th_range, C2_avg_th_dice_array, 'g', label="Mask Avg")
                plt.plot(th_range, C2_SI_th_dice_array, 'm', label="Mask SI")
                plt.xlim(0,1)
                plt.ylabel("Dice")
                plt.xlabel("Threshold")
                plt.axhline(SI_dice_C2,color='r',linestyle='--',label="BaseLine: DICE SI")
                plt.legend()
                plt.subplot(132)
                plt.title("recall x threshold")
                plt.plot(th_range, C2_union_th_recall_array, 'b', label="Mask Union")
                plt.plot(th_range, C2_avg_th_recall_array, 'g', label="Mask Avg")
                plt.plot(th_range, C2_SI_th_recall_array, 'm', label="Mask SI")
                plt.ylabel("Recall")
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.axhline(SI_recall_C2,color='r',linestyle='--',label="BaseLine: recall SI")
                plt.legend()
                plt.subplot(133)
                plt.title("precision x threshold")
                plt.plot(th_range, C2_union_th_precision_array, 'b', label="Mask Union")
                plt.plot(th_range, C2_avg_th_precision_array, 'g', label="Mask Avg")
                plt.plot(th_range, C2_SI_th_precision_array, 'm', label="Mask SI")
                plt.ylabel("Precision")
                plt.xlabel("Threshold")
                plt.xlim(0,1)
                plt.axhline(SI_precision_C2,color='r',linestyle='--',label="BaseLine: precision SI")
                plt.legend()
                plt.savefig(path_to_save_figure + image[:-4] + "_dicerecallprecision_x_threshold_C2_mean_then_BC.png")
                plt.close()
            
            C1_df_gen = pd.DataFrame.from_dict(C1_dict_max_values)
            C1_df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_C1_mean_then_BC.csv", index=True)
            
            C1_x = np.arange(0,len(C1_dict_max_values['th_union_max_dice']),1)
            
            plt.figure(figsize=(30,8))
            plt.suptitle("Union Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type + ", class: C1")
            
            plt.subplot(131)
            plt.title("Optimal Threshold for dice")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_union_max_dice']))
            plt.scatter(C1_x,C1_Y,color='b',label="Union Mask Optimal Threhsold - dice")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_union_dice'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_union_max_dice'])[np.where(np.array(C1_dict_max_values['flag_union_dice'])==True)]),color='r',
                        label='At this threshold, dice(th) is GREATER than baseline dice')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - dice")
            plt.legend()
            
            plt.subplot(132)
            plt.title("Optimal Threshold for recall")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_union_max_recall']))
            plt.scatter(C1_x,C1_Y,color='b',label="Union Mask Optimal Threhsold - recall")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_union_recall'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_union_max_recall'])[np.where(np.array(C1_dict_max_values['flag_union_recall'])==True)]),color='r',
                        label='At this threshold, recall(th) is GREATER than baseline recall')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - recall")
            plt.legend()
            
            plt.subplot(133)
            plt.title("Optimal Threshold for precision")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_union_max_precision']))
            plt.scatter(C1_x,C1_Y,color='b',label="Union Mask Optimal Threhsold - precision")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_union_precision'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_union_max_precision'])[np.where(np.array(C1_dict_max_values['flag_union_precision'])==True)]),color='r',
                        label='At this threshold, precision(th) is GREATER than baseline precision')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - precision")
            plt.legend()
            
            plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds_C1_mean_then_BC.png")
            plt.close()
            
            
            plt.figure(figsize=(30,8))
            plt.suptitle("avg Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type)
            
            plt.subplot(131)
            plt.title("Optimal Threshold for dice")
            Y = np.squeeze(np.array(C1_dict_max_values['th_avg_max_dice']))
            plt.scatter(C1_x,C1_Y,color='b',label="avg Mask Optimal Threhsold - dice")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_avg_dice'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_avg_max_dice'])[np.where(np.array(C1_dict_max_values['flag_avg_dice'])==True)]),
                        color='r',
                        label='At this threshold, dice(th) is GREATER than baseline dice')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - dice")
            plt.legend()
            
            plt.subplot(132)
            plt.title("Optimal Threshold for recall")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_avg_max_recall']))
            plt.scatter(C1_x,C1_Y,color='b',label="avg Mask Optimal Threhsold - recall")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_avg_recall'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_avg_max_recall'])[np.where(np.array(C1_dict_max_values['flag_avg_recall'])==True)]),
                        color='r',
                        label='At this threshold, recall(th) is GREATER than baseline recall')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - recall")
            plt.legend()
            
            plt.subplot(133)
            plt.title("Optimal Threshold for precision")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_avg_max_precision']))
            plt.scatter(C1_x,C1_Y,color='b',label="avg Mask Optimal Threhsold - precision")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_avg_precision'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_avg_max_precision'])[np.where(np.array(C1_dict_max_values['flag_avg_precision'])==True)]),
                        color='r',
                        label='At this threshold, precision(th) is GREATER than baseline precision')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - precision")
            plt.legend()
            
            plt.savefig(path_to_save_figure + "avg_mask_optimal_threhsolds_C1_mean_then_BC.png")
            plt.close()
            
            
            plt.figure(figsize=(30,8))
            plt.suptitle("SI Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type + ", class: C1")
            
            plt.subplot(131)
            plt.title("Optimal Threshold for dice")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_SI_max_dice']))
            plt.scatter(C1_x,C1_Y,color='b',label="SI Mask Optimal Threhsold - dice")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_SI_dice'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_SI_max_dice'])[np.where(np.array(C1_dict_max_values['flag_SI_dice'])==True)]),color='r',
                        label='At this threshold, dice(th) is GREATER than baseline dice')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - dice")
            plt.legend()
            
            plt.subplot(132)
            plt.title("Optimal Threshold for recall")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_SI_max_recall']))
            plt.scatter(C1_x,C1_Y,color='b',label="SI Mask Optimal Threhsold - recall")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_SI_recall'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_SI_max_recall'])[np.where(np.array(C1_dict_max_values['flag_SI_recall'])==True)]),color='r',
                        label='At this threshold, recall(th) is GREATER than baseline recall')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - recall")
            plt.legend()
            
            plt.subplot(133)
            plt.title("Optimal Threshold for precision")
            C1_Y = np.squeeze(np.array(C1_dict_max_values['th_SI_max_precision']))
            plt.scatter(C1_x,C1_Y,color='b',label="SI Mask Optimal Threhsold - precision")
            plt.scatter(C1_x[np.where(np.array(C1_dict_max_values['flag_SI_precision'])==True)],
                        np.squeeze(np.array(C1_dict_max_values['th_SI_max_precision'])[np.where(np.array(C1_dict_max_values['flag_SI_precision'])==True)]),color='r',
                        label='At this threshold, precision(th) is GREATER than baseline precision')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - precision")
            plt.legend()
            
            plt.savefig(path_to_save_figure + "SI_mask_optimal_threhsolds_C1_mean_then_BC.png")
            plt.close()
            
        
            C2_df_gen = pd.DataFrame.from_dict(C2_dict_max_values)
            C2_df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values_C2_mean_then_BC.csv", index=True)
            
            C2_x = np.arange(0,len(C2_dict_max_values['th_union_max_dice']),1)
            
            plt.figure(figsize=(30,8))
            plt.suptitle("Union Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type + ", class: C2")
            
            plt.subplot(131)
            plt.title("Optimal Threshold for dice")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_union_max_dice']))
            plt.scatter(C2_x,C2_Y,color='b',label="Union Mask Optimal Threhsold - dice")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_union_dice'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_union_max_dice'])[np.where(np.array(C2_dict_max_values['flag_union_dice'])==True)]),color='r',
                        label='At this threshold, dice(th) is GREATER than baseline dice')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - dice")
            plt.legend()
            
            plt.subplot(132)
            plt.title("Optimal Threshold for recall")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_union_max_recall']))
            plt.scatter(C2_x,C2_Y,color='b',label="Union Mask Optimal Threhsold - recall")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_union_recall'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_union_max_recall'])[np.where(np.array(C2_dict_max_values['flag_union_recall'])==True)]),color='r',
                        label='At this threshold, recall(th) is GREATER than baseline recall')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - recall")
            plt.legend()
            
            plt.subplot(133)
            plt.title("Optimal Threshold for precision")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_union_max_precision']))
            plt.scatter(C2_x,C2_Y,color='b',label="Union Mask Optimal Threhsold - precision")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_union_precision'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_union_max_precision'])[np.where(np.array(C2_dict_max_values['flag_union_precision'])==True)]),color='r',
                        label='At this threshold, precision(th) is GREATER than baseline precision')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - precision")
            plt.legend()
            
            plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds_C2_mean_then_BC.png")
            plt.close()
            
            
            plt.figure(figsize=(30,8))
            plt.suptitle("avg Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type)
            
            plt.subplot(131)
            plt.title("Optimal Threshold for dice")
            Y = np.squeeze(np.array(C2_dict_max_values['th_avg_max_dice']))
            plt.scatter(C2_x,C2_Y,color='b',label="avg Mask Optimal Threhsold - dice")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_avg_dice'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_avg_max_dice'])[np.where(np.array(C2_dict_max_values['flag_avg_dice'])==True)]),
                        color='r',
                        label='At this threshold, dice(th) is GREATER than baseline dice')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - dice")
            plt.legend()
            
            plt.subplot(132)
            plt.title("Optimal Threshold for recall")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_avg_max_recall']))
            plt.scatter(C2_x,C2_Y,color='b',label="avg Mask Optimal Threhsold - recall")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_avg_recall'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_avg_max_recall'])[np.where(np.array(C2_dict_max_values['flag_avg_recall'])==True)]),
                        color='r',
                        label='At this threshold, recall(th) is GREATER than baseline recall')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - recall")
            plt.legend()
            
            plt.subplot(133)
            plt.title("Optimal Threshold for precision")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_avg_max_precision']))
            plt.scatter(C2_x,C2_Y,color='b',label="avg Mask Optimal Threhsold - precision")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_avg_precision'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_avg_max_precision'])[np.where(np.array(C2_dict_max_values['flag_avg_precision'])==True)]),
                        color='r',
                        label='At this threshold, precision(th) is GREATER than baseline precision')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - precision")
            plt.legend()
            
            plt.savefig(path_to_save_figure + "avg_mask_optimal_threhsolds_C2_mean_then_BC.png")
            plt.close()
            
            
            plt.figure(figsize=(30,8))
            plt.suptitle("SI Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type + ", class: C2")
            
            plt.subplot(131)
            plt.title("Optimal Threshold for dice")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_SI_max_dice']))
            plt.scatter(C2_x,C2_Y,color='b',label="SI Mask Optimal Threhsold - dice")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_SI_dice'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_SI_max_dice'])[np.where(np.array(C2_dict_max_values['flag_SI_dice'])==True)]),color='r',
                        label='At this threshold, dice(th) is GREATER than baseline dice')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - dice")
            plt.legend()
            
            plt.subplot(132)
            plt.title("Optimal Threshold for recall")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_SI_max_recall']))
            plt.scatter(C2_x,C2_Y,color='b',label="SI Mask Optimal Threhsold - recall")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_SI_recall'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_SI_max_recall'])[np.where(np.array(C2_dict_max_values['flag_SI_recall'])==True)]),color='r',
                        label='At this threshold, recall(th) is GREATER than baseline recall')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - recall")
            plt.legend()
            
            plt.subplot(133)
            plt.title("Optimal Threshold for precision")
            C2_Y = np.squeeze(np.array(C2_dict_max_values['th_SI_max_precision']))
            plt.scatter(C2_x,C2_Y,color='b',label="SI Mask Optimal Threhsold - precision")
            plt.scatter(C2_x[np.where(np.array(C2_dict_max_values['flag_SI_precision'])==True)],
                        np.squeeze(np.array(C2_dict_max_values['th_SI_max_precision'])[np.where(np.array(C2_dict_max_values['flag_SI_precision'])==True)]),color='r',
                        label='At this threshold, precision(th) is GREATER than baseline precision')
            plt.xlabel("Image number")
            plt.ylabel("Opt th - precision")
            plt.legend()
            
            plt.savefig(path_to_save_figure + "SI_mask_optimal_threhsolds_C2_mean_then_BC.png")
            plt.close()