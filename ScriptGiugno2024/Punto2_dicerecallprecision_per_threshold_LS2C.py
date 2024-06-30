# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:16:43 2024

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

general_path_to_save = "D:/DATASET_Tesi_marzo2024_RESULTS_V11/"
general_dataset_path = "D:/DATASET_Tesi_marzo2024/" + dataset
general_results_path_2c = general_dataset_path + "k-net+swin/TEST_2classes/RESULTS"
general_results_path_3c = general_dataset_path + "k-net+swin/TEST_3classes/RESULTS"

#%%
for diff_type in tqdm(["PERT","MC"]):
    if diff_type == "PERT": diff_type_name = "_perturbation"
    if diff_type == "MC": diff_type_name = "_MC"
    for subset in tqdm(["test", "val"]):
        dict_max_values = {
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
            
            DIM = np.shape(GT_mask)[:2]
            softmax_matrix = wjb_functions.softmax_matrix_gen(softmax_path_2c, DIM, c, N)
            unc_map = wjb_functions.binary_entropy_map(softmax_matrix[:,:,1,:])
            
            mask_union = wjb_functions.mask_union_gen(diff_path_2c)
            mask_avg = wjb_functions.mask_avg_gen_2c(softmax_matrix)
            SI_mask = cv2.imread(SI_path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
            
            SI_dice = wjb_functions.dice(SI_mask,GT_mask)
            SI_recall = wjb_functions.recall(SI_mask,GT_mask)
            SI_precision = wjb_functions.precision(SI_mask,GT_mask)
            
            th_range = np.arange(1,11,1)/10
            
            union_th_dice_array = []
            avg_th_dice_array = []
            SI_th_dice_array = []
            
            union_th_recall_array = []
            avg_th_recall_array = []
            SI_th_recall_array = []
            
            union_th_precision_array = []
            avg_th_precision_array = []
            SI_th_precision_array = []
            
            for th in th_range:
                mask_th_union = mask_union & (unc_map<=th)
                mask_th_avg = mask_avg & (unc_map<=th)
                mask_th_SI = SI_mask & (unc_map<=th)
                
                # union_th_dice_array.append(wjb_functions.dice(GT_mask, mask_th_union))
                # avg_th_dice_array.append(wjb_functions.dice(GT_mask, mask_th_avg))
                # SI_th_dice_array.append(wjb_functions.dice(GT_mask, mask_th_SI))
                
                # union_th_recall_array.append(wjb_functions.recall(GT_mask, mask_th_union))
                # avg_th_recall_array.append(wjb_functions.recall(GT_mask, mask_th_avg))
                # SI_th_recall_array.append(wjb_functions.recall(GT_mask, mask_th_SI))
                
                # union_th_precision_array.append(wjb_functions.precision(GT_mask, mask_th_union))
                # avg_th_precision_array.append(wjb_functions.precision(GT_mask, mask_th_avg))
                # SI_th_precision_array.append(wjb_functions.precision(GT_mask, mask_th_SI))
                
                union_th_dice_array.append(wjb_functions.dice(mask_th_union, GT_mask))
                avg_th_dice_array.append(wjb_functions.dice(mask_th_avg, GT_mask))
                SI_th_dice_array.append(wjb_functions.dice(mask_th_SI, GT_mask))
                
                union_th_recall_array.append(wjb_functions.recall(mask_th_union,GT_mask))
                avg_th_recall_array.append(wjb_functions.recall(mask_th_avg,GT_mask))
                SI_th_recall_array.append(wjb_functions.recall(mask_th_SI,GT_mask))
                
                union_th_precision_array.append(wjb_functions.precision(mask_th_union,GT_mask))
                avg_th_precision_array.append(wjb_functions.precision(mask_th_avg,GT_mask))
                SI_th_precision_array.append(wjb_functions.precision(mask_th_SI,GT_mask))
                
                
            union_th_dice_array = np.array(union_th_dice_array)
            avg_th_dice_array = np.array(avg_th_dice_array)
            SI_th_dice_array = np.array(SI_th_dice_array)
            
            union_th_recall_array = np.array(union_th_recall_array)
            avg_th_recall_array = np.array(avg_th_recall_array)
            SI_th_recall_array = np.array(SI_th_recall_array)
            
            union_th_precision_array = np.array(union_th_precision_array)
            avg_th_precision_array = np.array(avg_th_precision_array)
            SI_th_precision_array = np.array(SI_th_precision_array)
            
            tabella_temp = np.array([union_th_dice_array,
                                     union_th_recall_array,
                                     union_th_precision_array,
                                     avg_th_dice_array,
                                     avg_th_recall_array,
                                     avg_th_precision_array,
                                     SI_th_dice_array,
                                     SI_th_recall_array,
                                     SI_th_precision_array
                                     ]).T
            
            df_temp = pd.DataFrame({'image': image,
                                    'union_th_dice_array': tabella_temp[:, 0], 
                                    'union_th_recall_array': tabella_temp[:, 1],
                                    'union_th_precision_array': tabella_temp[:, 2],
                                    'avg_th_dice_array': tabella_temp[:, 3],
                                    'avg_th_recall_array': tabella_temp[:, 4],
                                    'avg_th_precision_array': tabella_temp[:, 5],
                                    'SI_th_dice_array': tabella_temp[:, 6],
                                    'SI_th_recall_array': tabella_temp[:, 7],
                                    'SI_th_precision_array': tabella_temp[:, 8]
                                    })
            
            dict_max_values['image_name'].append(image)
            dict_max_values['union_max_dice'].append(np.nanmax(union_th_dice_array))
            dict_max_values['th_union_max_dice'].append(th_range[np.where(union_th_dice_array == np.nanmax(union_th_dice_array))[0][0]])
            dict_max_values['union_max_recall'].append(np.nanmax(union_th_recall_array))
            dict_max_values['th_union_max_recall'].append(th_range[np.where(union_th_recall_array == np.nanmax(union_th_recall_array))[0][0]])
            dict_max_values['union_max_precision'].append(np.nanmax(union_th_precision_array))
            dict_max_values['th_union_max_precision'].append(th_range[np.where(union_th_precision_array == np.nanmax(union_th_precision_array))[0][-1]])
            dict_max_values['avg_max_dice'].append(np.nanmax(avg_th_dice_array))
            dict_max_values['th_avg_max_dice'].append(th_range[np.where(avg_th_dice_array == np.nanmax(avg_th_dice_array))[0][0]])
            dict_max_values['avg_max_recall'].append(np.nanmax(avg_th_recall_array))
            dict_max_values['th_avg_max_recall'].append(th_range[np.where(avg_th_recall_array == np.nanmax(avg_th_recall_array))[0][0]])
            dict_max_values['avg_max_precision'].append(np.nanmax(avg_th_precision_array))
            if not avg_th_precision_array.all() == True:
                dict_max_values['th_avg_max_precision'].append(th_range[np.where(avg_th_precision_array == np.nanmax(avg_th_precision_array))[0][-1]])
            else: dict_max_values['th_avg_max_precision'].append(1)
            dict_max_values['SI_max_dice'].append(np.nanmax(SI_th_dice_array))
            dict_max_values['th_SI_max_dice'].append(th_range[np.where(SI_th_dice_array == np.nanmax(SI_th_dice_array))[0][0]])
            dict_max_values['SI_max_recall'].append(np.nanmax(SI_th_recall_array))
            dict_max_values['th_SI_max_recall'].append(th_range[np.where(SI_th_recall_array == np.nanmax(SI_th_recall_array))[0][0]])
            dict_max_values['SI_max_precision'].append(np.nanmax(SI_th_precision_array))
            dict_max_values['th_SI_max_precision'].append(th_range[np.where(SI_th_precision_array == np.nanmax(SI_th_precision_array))[0][-1]])
            
            if np.max(union_th_dice_array) > SI_dice: 
                dict_max_values['flag_union_dice'].append(1)
            else: 
                dict_max_values['flag_union_dice'].append(0)
            
            if np.max(union_th_recall_array) > SI_recall: 
                dict_max_values['flag_union_recall'].append(1)
            else: 
                dict_max_values['flag_union_recall'].append(0)
            
            if np.max(union_th_precision_array) > SI_precision: 
                dict_max_values['flag_union_precision'].append(1)
            else: 
                dict_max_values['flag_union_precision'].append(0)
                
            if np.max(avg_th_dice_array) > SI_dice: 
                dict_max_values['flag_avg_dice'].append(1)
            else: 
                dict_max_values['flag_avg_dice'].append(0)
            
            if np.max(avg_th_recall_array) > SI_recall: 
                dict_max_values['flag_avg_recall'].append(1)
            else: 
                dict_max_values['flag_avg_recall'].append(0)
            
            if np.max(avg_th_precision_array) > SI_precision: 
                dict_max_values['flag_avg_precision'].append(1)
            else: 
                dict_max_values['flag_avg_precision'].append(0)
                
            if np.max(SI_th_dice_array) > SI_dice: 
                dict_max_values['flag_SI_dice'].append(1)
            else: 
                dict_max_values['flag_SI_dice'].append(0)
            
            if np.max(SI_th_recall_array) > SI_recall: 
                dict_max_values['flag_SI_recall'].append(1)
            else: 
                dict_max_values['flag_SI_recall'].append(0)
            
            if np.max(SI_th_precision_array) > SI_precision: 
                dict_max_values['flag_SI_precision'].append(1)
            else: 
                dict_max_values['flag_SI_precision'].append(0)
            
            
            path_to_save_figure = general_path_to_save + dataset_name + diff_type + "/" + subset + "/"
            if not os.path.isdir(path_to_save_figure): os.makedirs(path_to_save_figure)
            
            df_temp.to_csv(path_to_save_figure + image[:-4] + "_DATAFRAME.csv", index = False)
            
            plt.figure(figsize=(40,8))
            plt.suptitle("Image " + image + " from dataset " + dataset[:-1] + ", subset: " + subset)
            plt.subplot(131)
            plt.title("dice x threshold")
            plt.plot(th_range, union_th_dice_array, 'b', label="Mask Union")
            plt.plot(th_range, avg_th_dice_array, 'g', label="Mask Avg")
            plt.plot(th_range, SI_th_dice_array, 'm', label="Mask SI")
            plt.xlim(0,1)
            plt.ylabel("Dice")
            plt.xlabel("Threshold")
            plt.axhline(SI_dice,color='r',linestyle='--',label="BaseLine: DICE SI")
            plt.legend()
            plt.subplot(132)
            plt.title("recall x threshold")
            plt.plot(th_range, union_th_recall_array, 'b', label="Mask Union")
            plt.plot(th_range, avg_th_recall_array, 'g', label="Mask Avg")
            plt.plot(th_range, SI_th_recall_array, 'm', label="Mask SI")
            plt.ylabel("Recall")
            plt.xlabel("Threshold")
            plt.xlim(0,1)
            plt.axhline(SI_recall,color='r',linestyle='--',label="BaseLine: recall SI")
            plt.legend()
            plt.subplot(133)
            plt.title("precision x threshold")
            plt.plot(th_range, union_th_precision_array, 'b', label="Mask Union")
            plt.plot(th_range, avg_th_precision_array, 'g', label="Mask Avg")
            plt.plot(th_range, SI_th_precision_array, 'm', label="Mask SI")
            plt.ylabel("Precision")
            plt.xlabel("Threshold")
            plt.xlim(0,1)
            plt.axhline(SI_precision,color='r',linestyle='--',label="BaseLine: precision SI")
            plt.legend()
            plt.savefig(path_to_save_figure + image[:-4] + "_dicerecallprecision_x_threshold.png")
            plt.close()
        
        df_gen = pd.DataFrame.from_dict(dict_max_values)
        df_gen.to_csv(path_to_save_figure + "DATAFRAME_max_values.csv", index=True)
        
        x = np.arange(0,len(dict_max_values['th_union_max_dice']),1)
        
        plt.figure(figsize=(40,8))
        plt.suptitle("Union Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type)
        
        plt.subplot(131)
        plt.title("Optimal Threshold for dice")
        Y = np.squeeze(np.array(dict_max_values['th_union_max_dice']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_union_dice'])==True)],
                    np.squeeze(np.array(dict_max_values['th_union_max_dice'])[np.where(np.array(dict_max_values['flag_union_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - dice")
        plt.legend()
        
        plt.subplot(132)
        plt.title("Optimal Threshold for recall")
        Y = np.squeeze(np.array(dict_max_values['th_union_max_recall']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - recall")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_union_recall'])==True)],
                    np.squeeze(np.array(dict_max_values['th_union_max_recall'])[np.where(np.array(dict_max_values['flag_union_recall'])==True)]),color='r',
                    label='At this threshold, recall(th) is GREATER than baseline recall')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - recall")
        plt.legend()
        
        plt.subplot(133)
        plt.title("Optimal Threshold for precision")
        Y = np.squeeze(np.array(dict_max_values['th_union_max_precision']))
        plt.scatter(x,Y,color='b',label="Union Mask Optimal Threhsold - precision")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_union_precision'])==True)],
                    np.squeeze(np.array(dict_max_values['th_union_max_precision'])[np.where(np.array(dict_max_values['flag_union_precision'])==True)]),color='r',
                    label='At this threshold, precision(th) is GREATER than baseline precision')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - precision")
        plt.legend()
        
        plt.savefig(path_to_save_figure + "union_mask_optimal_threhsolds.png")
        plt.close()
        
        
        plt.figure(figsize=(40,8))
        plt.suptitle("avg Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type)
        
        plt.subplot(131)
        plt.title("Optimal Threshold for dice")
        Y = np.squeeze(np.array(dict_max_values['th_avg_max_dice']))
        plt.scatter(x,Y,color='b',label="avg Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_avg_dice'])==True)],
                    np.squeeze(np.array(dict_max_values['th_avg_max_dice'])[np.where(np.array(dict_max_values['flag_avg_dice'])==True)]),
                    color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - dice")
        plt.legend()
        
        plt.subplot(132)
        plt.title("Optimal Threshold for recall")
        Y = np.squeeze(np.array(dict_max_values['th_avg_max_recall']))
        plt.scatter(x,Y,color='b',label="avg Mask Optimal Threhsold - recall")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_avg_recall'])==True)],
                    np.squeeze(np.array(dict_max_values['th_avg_max_recall'])[np.where(np.array(dict_max_values['flag_avg_recall'])==True)]),
                    color='r',
                    label='At this threshold, recall(th) is GREATER than baseline recall')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - recall")
        plt.legend()
        
        plt.subplot(133)
        plt.title("Optimal Threshold for precision")
        Y = np.squeeze(np.array(dict_max_values['th_avg_max_precision']))
        plt.scatter(x,Y,color='b',label="avg Mask Optimal Threhsold - precision")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_avg_precision'])==True)],
                    np.squeeze(np.array(dict_max_values['th_avg_max_precision'])[np.where(np.array(dict_max_values['flag_avg_precision'])==True)]),
                    color='r',
                    label='At this threshold, precision(th) is GREATER than baseline precision')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - precision")
        plt.legend()
        
        plt.savefig(path_to_save_figure + "avg_mask_optimal_threhsolds.png")
        plt.close()
        
        
        plt.figure(figsize=(40,8))
        plt.suptitle("SI Mask: Dice Recall Precision - dataset: " + dataset[:-1] + " , subset: " + subset + ", " + diff_type)
        
        plt.subplot(131)
        plt.title("Optimal Threshold for dice")
        Y = np.squeeze(np.array(dict_max_values['th_SI_max_dice']))
        plt.scatter(x,Y,color='b',label="SI Mask Optimal Threhsold - dice")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_SI_dice'])==True)],
                    np.squeeze(np.array(dict_max_values['th_SI_max_dice'])[np.where(np.array(dict_max_values['flag_SI_dice'])==True)]),color='r',
                    label='At this threshold, dice(th) is GREATER than baseline dice')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - dice")
        plt.legend()
        
        plt.subplot(132)
        plt.title("Optimal Threshold for recall")
        Y = np.squeeze(np.array(dict_max_values['th_SI_max_recall']))
        plt.scatter(x,Y,color='b',label="SI Mask Optimal Threhsold - recall")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_SI_recall'])==True)],
                    np.squeeze(np.array(dict_max_values['th_SI_max_recall'])[np.where(np.array(dict_max_values['flag_SI_recall'])==True)]),color='r',
                    label='At this threshold, recall(th) is GREATER than baseline recall')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - recall")
        plt.legend()
        
        plt.subplot(133)
        plt.title("Optimal Threshold for precision")
        Y = np.squeeze(np.array(dict_max_values['th_SI_max_precision']))
        plt.scatter(x,Y,color='b',label="SI Mask Optimal Threhsold - precision")
        plt.scatter(x[np.where(np.array(dict_max_values['flag_SI_precision'])==True)],
                    np.squeeze(np.array(dict_max_values['th_SI_max_precision'])[np.where(np.array(dict_max_values['flag_SI_precision'])==True)]),color='r',
                    label='At this threshold, precision(th) is GREATER than baseline precision')
        plt.xlabel("Image number")
        plt.ylabel("Opt th - precision")
        plt.legend()
        
        plt.savefig(path_to_save_figure + "SI_mask_optimal_threhsolds.png")
        plt.close()