# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:25:40 2024

@author: willy
"""

#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Img
from tqdm import tqdm

#%% FUNCTIONS

def std_map(softmax_matrix): return np.nanstd(softmax_matrix, axis=2)

def binary_entropy_map(softmax_matrix):
    p_mean = np.mean(softmax_matrix, axis=2)
    p_mean[p_mean==0] = 1e-8 
    p_mean[p_mean==1] = 1-1e-8
    HB_pert = -(np.multiply(p_mean,np.log2(p_mean)) + np.multiply((1-p_mean),np.log2(1-p_mean)))
    return HB_pert

def shannon_entropy_map(softmax_matrix):
    shape = np.shape(softmax_matrix)
    shannon_entropy_map = np.zeros([shape[0],shape[1]]).astype(float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ij_vector = softmax_matrix[i,j,:]
            shannon_entropy_map[i,j] = -(np.multiply(ij_vector[ij_vector!=0], np.log2(ij_vector[ij_vector!=0]))).sum()
    return shannon_entropy_map

def maxminscal(array):
    MASSIMO = np.nanmax(array)
    minimo = np.nanmin(array)
    return (array-minimo)/(MASSIMO-minimo)

def dice(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()

    if TP==0 and FN==0 and FP==0:
        dice_ind = np.nan
    else:
        dice_ind = 2*TP/(2*TP+FP+FN)
    return dice_ind

def mean_ent(ent_map_matrix): return np.nanmean(ent_map_matrix)
def mean_std(std_map_matrix): return np.nanmean(std_map_matrix)

def mean_ent_multiclass(ent_map_matrix): return np.nanmean(ent_map_matrix[:,:,1:])
def mean_std_multiclass(std_map_matrix): return np.nanmean(std_map_matrix[:,:,1:])

def mean_dice_mat(dice_mat_temp): return np.nanmean(dice_mat_temp)
def std_dice_mat(dice_mat_temp): return np.nanstd(dice_mat_temp)
#%%
total_path = 'D:/DATASET_Tesi_marzo2024/'
result_path = 'D:/DATASET_Tesi_marzo2024_RESULTS/'
N = 20 #numero sample montecarlo / perturbation

#%%
dict_of_dataset = {
    
    'LS2c' : {
            'path': "Liver HE steatosis/DATASET_2classes/",
            'test': "Liver HE steatosis/k-net+swin/TEST_2classes/",
            'c' : 2
            },
    
    'LS3c' : {
            'path': "Liver HE steatosis/DATASET_3classes/",
            'test': "Liver HE steatosis/k-net+swin/TEST_3classes/",
            'c' : 3
            },
    
    'RG3c'  : {
            'path': "Renal PAS glomeruli/DATASET/",
            'test': "Renal PAS glomeruli/k-net+swin/TEST/",
            'c' : 3
            },
    
    'RT3c'  : {
            'path': "Renal PAS tubuli/DATASET/",
            'test': "Renal PAS tubuli/k-net+swin/TEST/",
            'c' : 3
            }
}

#%%
for dataset in dict_of_dataset:
    c = dict_of_dataset[dataset]['c']
    for subset in ['train','val','test']:
        
        metrics_per_subset = pd.DataFrame(columns=[
            'image_name',
            'mean_bin_ent_MC',
            'mean_bin_ent_MC_1',
            'mean_bin_ent_MC_2',
            'mean_sha_ent_MC',
            'mean_sha_ent_MC_1',
            'mean_sha_ent_MC_2',
            'mean_std_MC',
            'mean_std_MC_1',
            'mean_std_MC_2',
            'mean_dice_mat_MC',
            'mean_dice_mat_MC_1',
            'mean_dice_mat_MC_2',
            'std_dice_mat_MC',
            'std_dice_mat_MC_1',
            'std_dice_mat_MC_2',
            'mean_bin_ent_PERT',
            'mean_bin_ent_PERT_1',
            'mean_bin_ent_PERT_2',
            'mean_sha_ent_PERT',
            'mean_sha_ent_PERT_1',
            'mean_sha_ent_PERT_2',
            'mean_std_PERT',
            'mean_std_PERT_1',
            'mean_std_PERT_2',
            'mean_dice_mat_PERT',
            'mean_dice_mat_PERT_1',
            'mean_dice_mat_PERT_2',
            'std_dice_mat_PERT',
            'std_dice_mat_PERT_1',
            'std_dice_mat_PERT_2'
            ])
        
        GT_mask_path = total_path+dict_of_dataset[dataset]['path']+subset+"/manual/"

        image_list = os.listdir(GT_mask_path)
        
        for name in tqdm(image_list):
            GT_mask = np.array(Img.open(GT_mask_path+name))/255
            DIM = np.shape(GT_mask)
            
            MC_mask_path = total_path + dict_of_dataset[dataset]['test'] + "RESULTS_MC/" + subset + "/mask/" + name + "/"
            MC_softmax_path = total_path + dict_of_dataset[dataset]['test'] + "RESULTS_MC/" + subset + "/softmax/" + name + "/"
            PERT_mask_path = total_path + dict_of_dataset[dataset]['test'] + "RESULTS_perturbation/" + subset + "/mask/" + name + "/"
            PERT_softmax_path = total_path + dict_of_dataset[dataset]['test'] + "RESULTS_perturbation/" + subset + "/softmax/" + name + "/"
            
            # MC_result_maps_path = result_path + dict_of_dataset[dataset]['test'] + "result_maps_MC/" + subset + "/" + name + "/"
            # PERT_result_maps_path = result_path + dict_of_dataset[dataset]['test'] + "result_maps_PERT/" + subset + "/" + name + "/"
            
            result_maps_path = result_path + dict_of_dataset[dataset]['test'] + subset + "/" + name + "/"
            if not os.path.isdir(result_maps_path): os.makedirs(result_maps_path)
            
            # if not os.path.isdir(MC_result_maps_path): os.makedirs(MC_result_maps_path)
            # if not os.path.isdir(PERT_result_maps_path): os.makedirs(PERT_result_maps_path)
            
            ########## creation softmax e seg matrix
     
            # Creo softmax_matrix per MonteCarlo
            softmax_matrix_MC = np.zeros((DIM[0],DIM[1],c,N),dtype=np.float32)
            counter_MC = 0
            for num in os.listdir(MC_softmax_path):
                for n_class in range(c):
                    st0 = np.float32((np.load(MC_softmax_path + "/" + num)['softmax'])[:,:,n_class])
                    softmax_matrix_MC[:,:,n_class,counter_MC] = np.copy(st0)
                counter_MC += 1
            
                
            # Creo softmax_matrix per perturbation            
            softmax_matrix_PERT = np.zeros((DIM[0],DIM[1],c,N),dtype=np.float32)
            counter_PERT = 0
            for num in os.listdir(PERT_softmax_path):
                for n_class in range(c):
                    st0 = np.float32((np.load(PERT_softmax_path + "/" + num)['softmax'])[:,:,n_class])
                    softmax_matrix_PERT[:,:,n_class,counter_PERT] = np.copy(st0)
                counter_PERT += 1

            # Creo seg_matrix per MonteCarlo            
            seg_matrix_MC = np.zeros((DIM[0],DIM[1],N))
            counter = 0
            for num in os.listdir(MC_mask_path):
                temp = Img.open(MC_mask_path + num)
                seg_matrix_MC[:,:,counter] = np.array(temp)/255
                counter += 1
                
            # Creo seg_matrix per perturbation            
            seg_matrix_PERT = np.zeros((DIM[0],DIM[1],N))
            counter = 0
            for num in os.listdir(PERT_mask_path):
                temp = Img.open(PERT_mask_path + num)
                seg_matrix_PERT[:,:,counter] = np.array(temp)/255
                counter += 1
            
            ################ maps creation
            
            # Creo std_map MC
            std_map_matrix_MC = np.zeros([DIM[0],DIM[1],c])
            for n_class in range(c):
                std_map_matrix_MC[:,:,n_class] = std_map(softmax_matrix_MC[:,:,n_class,:])
            
            np.save(result_maps_path + "std_map_matrix_MC",std_map_matrix_MC)
            
            # Creo std_map perturbations
            std_map_matrix_PERT = np.zeros([DIM[0],DIM[1],c])
            for n_class in range(c):
                std_map_matrix_PERT[:,:,n_class] = std_map(softmax_matrix_PERT[:,:,n_class,:])
            
            np.save(result_maps_path + "std_map_matrix_PERT",std_map_matrix_PERT)
            
            # Creo binary_ent_map MonteCarlo
            binary_ent_map_matrix_MC = np.zeros([DIM[0],DIM[1],c])
            for n_class in range(c):
                binary_ent_map_matrix_MC[:,:,n_class] = binary_entropy_map(softmax_matrix_MC[:,:,n_class,:])
            
            np.save(result_maps_path + "binary_ent_map_matrix_MC",binary_ent_map_matrix_MC)
                
            # Creo binary_ent_map perturbation
            binary_ent_map_matrix_PERT = np.zeros([DIM[0],DIM[1],c])
            for n_class in range(c):
                binary_ent_map_matrix_PERT[:,:,n_class] = binary_entropy_map(softmax_matrix_PERT[:,:,n_class,:])
                
            np.save(result_maps_path + "binary_ent_map_matrix_PERT",binary_ent_map_matrix_PERT)
            
            # Creo shannon_ent_map MonteCarlo
            shannon_ent_map_matrix_MC = np.zeros([DIM[0],DIM[1],c])
            for n_class in range(c):
                shannon_ent_map_matrix_MC[:,:,n_class] = shannon_entropy_map(softmax_matrix_MC[:,:,n_class,:])
            
            np.save(result_maps_path + "shannon_ent_map_matrix_MC",shannon_ent_map_matrix_MC)
                
            # Creo shannon_ent_map perturbation
            shannon_ent_map_matrix_PERT = np.zeros([DIM[0],DIM[1],c])
            for n_class in range(c):
                shannon_ent_map_matrix_PERT[:,:,n_class] = shannon_entropy_map(softmax_matrix_PERT[:,:,n_class,:])
                
            np.save(result_maps_path + "shannon_ent_map_matrix_PERT",shannon_ent_map_matrix_PERT)
            
            if c == 3:
                # Creo dice_mat MC
                dice_mat_MC = -np.ones((N,N,c))
                # dice_array_GT_MC = []
                for i in range(N):
                    for j in range(i+1,N):
                        seg_i = seg_matrix_MC[:,:,i]
                        seg_j = seg_matrix_MC[:,:,j]
                        
                        max_seg = np.unique(seg_i)[-1]
                        med_seg = np.unique(seg_i)[-2]
                        
                        seg_i1 = np.copy(seg_i)
                        seg_i1[seg_i==max_seg] = 1
                        seg_i1[seg_i<max_seg] = 0
                        
                        seg_j1 = np.copy(seg_j)
                        seg_j1[seg_j==max_seg] = 1
                        seg_j1[seg_j<max_seg] = 0
                        
                        seg_i2 = np.copy(seg_i)
                        seg_i2[seg_i==med_seg] = 1
                        seg_i2[seg_i>med_seg] = 0
                        
                        seg_j2 = np.copy(seg_j)
                        seg_j2[seg_j==med_seg] = 1
                        seg_j2[seg_j>med_seg] = 0
                        
                        dice_mat_MC[i,j,0] = dice(seg_i1,seg_j1)
                        dice_mat_MC[i,j,1] = dice(seg_i2,seg_j2)
                    # dice_array_GT_MC.append(dice(seg_matrix_MC[:,:,i],GT_mask))
                dice_mat_MC[dice_mat_MC<0] = np.nan
                
                np.save(result_maps_path + "dice_mat_MC",dice_mat_MC)
                
                # Creo dice_mat PERT
                dice_mat_PERT = -np.ones((N,N,c))
                # dice_array_GT_PERT = []
                for i in range(N):
                    for j in range(i+1,N):
                        seg_i = seg_matrix_PERT[:,:,i]
                        seg_j = seg_matrix_PERT[:,:,j]
                        max_seg = np.unique(seg_i)[-1]
                        med_seg = np.unique(seg_i)[-2]
                        
                        seg_i1 = np.copy(seg_i)
                        seg_i1[seg_i==max_seg] = 1
                        seg_i1[seg_i<max_seg] = 0
                        
                        seg_j1 = np.copy(seg_j)
                        seg_j1[seg_j==max_seg] = 1
                        seg_j1[seg_j<max_seg] = 0
                        
                        seg_i2 = np.copy(seg_i)
                        seg_i2[seg_i==med_seg] = 1
                        seg_i2[seg_i>med_seg] = 0
                        
                        seg_j2 = np.copy(seg_j)
                        seg_j2[seg_j==med_seg] = 1
                        seg_j2[seg_j>med_seg] = 0
                        
                        dice_mat_PERT[i,j,0] = dice(seg_i1,seg_j1)
                        dice_mat_PERT[i,j,1] = dice(seg_i2,seg_j2)
                    # dice_array_GT_PERT.append(dice(seg_matrix_PERT[:,:,i],GT_mask))
                dice_mat_PERT[dice_mat_PERT<0] = np.nan
                
                
            if c == 2:
                # Creo dice_mat MC
                dice_mat_MC = -np.ones((N,N))
                dice_array_GT_MC = []
                for i in range(N):
                    for j in range(i+1,N):
                        dice_mat_MC[i,j] = dice(seg_matrix_MC[:,:,i],seg_matrix_MC[:,:,j])
                    dice_array_GT_MC.append(dice(seg_matrix_MC[:,:,i],GT_mask))
                dice_mat_MC[dice_mat_MC<0] = np.nan
                
                np.save(result_maps_path + "dice_mat_MC",dice_mat_MC)
                
                # Creo dice_mat PERT
                dice_mat_PERT = -np.ones((N,N))
                dice_array_GT_PERT = []
                for i in range(N):
                    for j in range(i+1,N):
                        dice_mat_PERT[i,j] = dice(seg_matrix_PERT[:,:,i],seg_matrix_PERT[:,:,j])
                    dice_array_GT_PERT.append(dice(seg_matrix_PERT[:,:,i],GT_mask))
                dice_mat_PERT[dice_mat_PERT<0] = np.nan
                
                np.save(result_maps_path + "dice_mat_PERT",dice_mat_PERT)
            
            ############# metrics calculation on the maps
            if c == 2:
                
                # binary ent MC
                mean_bin_ent_MC = mean_ent(binary_ent_map_matrix_MC[:,:,1])
                
                # shannon ent MC
                mean_sha_ent_MC = mean_ent(shannon_ent_map_matrix_MC[:,:,1])
                
                # std MC
                mean_std_MC = mean_std(std_map_matrix_MC[:,:,1])
                
                # dice mat MC
                mean_dice_mat_MC = mean_dice_mat(dice_mat_MC)
                std_dice_mat_MC = std_dice_mat(dice_mat_MC)
                
                
                # binary ent PERT
                mean_bin_ent_PERT = mean_ent(binary_ent_map_matrix_PERT[:,:,1])
                
                # shannon ent PERT
                mean_sha_ent_PERT = mean_ent(shannon_ent_map_matrix_PERT[:,:,1])
                
                # std PERT
                mean_std_PERT = mean_std(std_map_matrix_PERT[:,:,1])
                
                # dice mat PERT
                mean_dice_mat_PERT = mean_dice_mat(dice_mat_PERT)
                std_dice_mat_PERT = std_dice_mat(dice_mat_PERT)
                
                row_metrics = {
                'image_name' : name,
                'mean_bin_ent_MC': mean_bin_ent_MC,
                'mean_bin_ent_MC_1':np.nan,
                'mean_bin_ent_MC_2':np.nan,
                'mean_sha_ent_MC':mean_sha_ent_MC,
                'mean_sha_ent_MC_1':np.nan,
                'mean_sha_ent_MC_2':np.nan,
                'mean_std_MC':mean_std_MC,
                'mean_std_MC_1':np.nan,
                'mean_std_MC_2':np.nan,
                'mean_dice_mat_MC':mean_dice_mat_MC,
                'mean_dice_mat_MC_1':np.nan,
                'mean_dice_mat_MC_2':np.nan,
                'std_dice_mat_MC':std_dice_mat_MC,
                'std_dice_mat_MC_1':np.nan,
                'std_dice_mat_MC_2':np.nan,
                'mean_bin_ent_PERT':mean_bin_ent_PERT,
                'mean_bin_ent_PERT_1':np.nan,
                'mean_bin_ent_PERT_2':np.nan,
                'mean_sha_ent_PERT':mean_sha_ent_PERT,
                'mean_sha_ent_PERT_1':np.nan,
                'mean_sha_ent_PERT_2':np.nan,
                'mean_std_PERT':mean_std_PERT,
                'mean_std_PERT_1':np.nan,
                'mean_std_PERT_2':np.nan,
                'mean_dice_mat_PERT':mean_dice_mat_PERT,
                'mean_dice_mat_PERT_1':np.nan,
                'mean_dice_mat_PERT_2':np.nan,
                'std_dice_mat_PERT':std_dice_mat_PERT,
                'std_dice_mat_PERT_1':np.nan,
                'std_dice_mat_PERT_2':np.nan
                }
                
            if c == 3:
            
                # binary ent MC
                mean_bin_ent_MC = mean_ent_multiclass(binary_ent_map_matrix_MC[:,:,1:])
                mean_bin_ent_MC_1 = mean_ent(binary_ent_map_matrix_MC[:,:,1])
                mean_bin_ent_MC_2 = mean_ent(binary_ent_map_matrix_MC[:,:,2])
                
                # shannon ent MC
                mean_sha_ent_MC = mean_ent_multiclass(shannon_ent_map_matrix_MC[:,:,1:])
                mean_sha_ent_MC_1 = mean_ent(shannon_ent_map_matrix_MC[:,:,1])
                mean_sha_ent_MC_2 = mean_ent(shannon_ent_map_matrix_MC[:,:,2])
                
                # std MC
                mean_std_MC = mean_std_multiclass(std_map_matrix_MC[:,:,1:])
                mean_std_MC_1 = mean_std(std_map_matrix_MC[:,:,1])
                mean_std_MC_2 = mean_std(std_map_matrix_MC[:,:,2])
                
                # dice mat MC
                mean_dice_mat_MC = mean_dice_mat(dice_mat_MC)
                std_dice_mat_MC = std_dice_mat(dice_mat_MC)
                
                mean_dice_mat_MC_1 = mean_dice_mat(dice_mat_MC[:,:,0])
                mean_dice_mat_MC_2 = mean_dice_mat(dice_mat_MC[:,:,1])
                std_dice_mat_MC_1 = std_dice_mat(dice_mat_MC[:,:,0])
                std_dice_mat_MC_2 = std_dice_mat(dice_mat_MC[:,:,1])
                
                
                # binary ent PERT
                mean_bin_ent_PERT = mean_ent_multiclass(binary_ent_map_matrix_PERT[:,:,1:])
                mean_bin_ent_PERT_1 = mean_ent(binary_ent_map_matrix_PERT[:,:,1])
                mean_bin_ent_PERT_2 = mean_ent(binary_ent_map_matrix_PERT[:,:,2])
                
                # shannon ent PERT
                mean_sha_ent_PERT = mean_ent_multiclass(shannon_ent_map_matrix_PERT[:,:,1:])
                mean_sha_ent_PERT_1 = mean_ent(shannon_ent_map_matrix_PERT[:,:,1])
                mean_sha_ent_PERT_2 = mean_ent(shannon_ent_map_matrix_PERT[:,:,2])
                
                # std PERT
                mean_std_PERT = mean_std_multiclass(std_map_matrix_PERT[:,:,1:])
                mean_std_PERT_1 = mean_std(std_map_matrix_PERT[:,:,1])
                mean_std_PERT_2 = mean_std(std_map_matrix_PERT[:,:,2])
                
                # dice mat PERT
                mean_dice_mat_PERT = mean_dice_mat(dice_mat_PERT)
                std_dice_mat_PERT = std_dice_mat(dice_mat_PERT)
                
                mean_dice_mat_PERT_1 = mean_dice_mat(dice_mat_PERT[:,:,0])
                mean_dice_mat_PERT_2 = mean_dice_mat(dice_mat_PERT[:,:,1])
                std_dice_mat_PERT_1 = std_dice_mat(dice_mat_PERT[:,:,0])
                std_dice_mat_PERT_2 = std_dice_mat(dice_mat_PERT[:,:,1])
                
                row_metrics = {
                'image_name' : name,
                'mean_bin_ent_MC': mean_bin_ent_MC,
                'mean_bin_ent_MC_1':mean_bin_ent_MC_1,
                'mean_bin_ent_MC_2':mean_bin_ent_MC_2,
                'mean_sha_ent_MC':mean_sha_ent_MC,
                'mean_sha_ent_MC_1':mean_sha_ent_MC_1,
                'mean_sha_ent_MC_2':mean_sha_ent_MC_2,
                'mean_std_MC':mean_std_MC,
                'mean_std_MC_1':mean_std_MC_1,
                'mean_std_MC_2':mean_std_MC_2,
                'mean_dice_mat_MC':mean_dice_mat_MC,
                'mean_dice_mat_MC_1':mean_dice_mat_MC_1,
                'mean_dice_mat_MC_2':mean_dice_mat_MC_2,
                'std_dice_mat_MC':std_dice_mat_MC,
                'std_dice_mat_MC_1':std_dice_mat_MC_1,
                'std_dice_mat_MC_2':std_dice_mat_MC_2,
                'mean_bin_ent_PERT':mean_bin_ent_PERT,
                'mean_bin_ent_PERT_1':mean_bin_ent_PERT_1,
                'mean_bin_ent_PERT_2':mean_bin_ent_PERT_2,
                'mean_sha_ent_PERT':mean_sha_ent_PERT,
                'mean_sha_ent_PERT_1':mean_sha_ent_PERT_1,
                'mean_sha_ent_PERT_2':mean_sha_ent_PERT_2,
                'mean_std_PERT':mean_std_PERT,
                'mean_std_PERT_1':mean_std_PERT_1,
                'mean_std_PERT_2':mean_std_PERT_2,
                'mean_dice_mat_PERT':mean_dice_mat_PERT,
                'mean_dice_mat_PERT_1':mean_dice_mat_PERT_1,
                'mean_dice_mat_PERT_2':mean_dice_mat_PERT_2,
                'std_dice_mat_PERT':std_dice_mat_PERT,
                'std_dice_mat_PERT_1':std_dice_mat_PERT_1,
                'std_dice_mat_PERT_2':std_dice_mat_PERT_2
                }
            
            metrics_per_subset = pd.concat([metrics_per_subset, pd.DataFrame(row_metrics,index=[0])])
            
    path_to_save_metrics_per_subset = result_path + dict_of_dataset[dataset]['test'] + subset
    metrics_per_subset.to_csv(path_to_save_metrics_per_subset+"/metrics.csv",index=False)
    #         break
    #     break
    # break
        
        
        
        
        
        
        
        