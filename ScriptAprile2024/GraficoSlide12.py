# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:46:28 2024

@author: willy
"""

# CODICE BASE PER PROVARE

#%% IMPORT
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Img
from tqdm import tqdm

#%% PRIMARY FUNCTIONS
def dice(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()

    if TP==0 and FN==0 and FP==0:
        dice_ind = np.nan
    else:
        dice_ind = 2*TP/(2*TP+FP+FN)
    return dice_ind

def precision(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()

    if TP==0 and FN==0 and FP==0:
        precision_ind = np.nan
    else:
        precision_ind = TP/(TP+FP)
    return precision_ind

def recall(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()

    if TP==0 and FN==0 and FP==0:
        recall_ind = np.nan
    else:
        recall_ind = TP/(TP+FN)
    return recall_ind

def mask_splitter(mask):
    mask_C1 = np.copy(mask)
    mask_C2 = np.copy(mask)

    mask_C1[mask>0.8]=0
    mask_C1[mask>0]=1
    mask_C2[mask<0.8]=0
    mask_C2[mask>0]=1
    return mask_C1, mask_C2

def mask_generator_for_calcoli(SI_mask,uncert_map,Th,softmax_matrix): #da fare una volta per classe
    
    mean_softmax = np.mean(softmax_matrix,axis=-1)    
    mask_avg = np.argmax(mean_softmax,axis=-1)
    
    mask_uncert = uncert_map > Th   #(0.1, 0.2, 0.3, ...)
    mask_cert = (~mask_uncert) & mask_avg
    
    mask_FP = np.copy(mask_uncert)
    mask_FN = np.copy(mask_cert)
    
    mask_auto = np.copy(SI_mask)
    mask_auto[mask_FP] = False
    mask_auto[mask_FN] = True
    
    return mask_auto
    #Calcolo di Dice, Precison, Recall tra "GT" e "mask_auto"

#%% MEGAMATRIX GENERATION
# Softmax matrix generator
def softmax_matrix_gen(softmax_path,DIM,c,N):
    softmax_matrix = np.zeros((DIM[0],DIM[1],c,N),dtype=np.float32)
    counter = 0
    for num in os.listdir(softmax_path):
        for n_class in range(c):
            st0 = np.float32((np.load(softmax_path + "/" + num)['softmax'])[:,:,n_class])
            softmax_matrix[:,:,n_class,counter] = np.copy(st0)
        counter += 1
    return softmax_matrix
        
# Seg Matrix generator
def seg_matrix_gen(mask_path,DIM,N):
    seg_matrix = np.zeros((DIM[0],DIM[1],N))
    counter = 0
    for num in os.listdir(mask_path):
        temp = Img.open(mask_path + num)
        seg_matrix[:,:,counter] = np.array(temp)/255
        counter += 1
    values = np.unique(seg_matrix)
    classes = len(values)
    standard_values = [0,0.5,1]
    for i in range(classes):
        seg_matrix[seg_matrix==values[i]] = standard_values[i]
    if np.max(seg_matrix)==0.5: seg_matrix[seg_matrix==0.5] = 1
    return seg_matrix

#%% UNCERTAINTY MAP GENERATION
#%% METRICS CALCULATOR
# Binary Entropy Map
def binary_entropy_map(softmax_matrix):
    p_mean = np.mean(softmax_matrix, axis=2)
    p_mean[p_mean==0] = 1e-8 
    p_mean[p_mean==1] = 1-1e-8
    HB_pert = -(np.multiply(p_mean,np.log2(p_mean)) + np.multiply((1-p_mean),np.log2(1-p_mean)))
    return HB_pert
#%% PLOT GENERATOR
def boxplot_generator(dice_GTSI, dice_MC, dice_PERT, TOT_dice, fig_save_path, name, dataset, subset, c):
    data = [dice_GTSI, dice_MC, dice_PERT, TOT_dice]
    fig = plt.figure(figsize =(10, 7))
    x_label = ['Single Inference Dice', 'Monte Carlo Dices', 'Perturbations Dices', 'Combined Dices']

    bp = plt.boxplot(data)
    plt.title("Image " + name[-4] + " of " + dataset + " (subset " + subset + ") - class " + c)
    plt.ylabel('DICE')
    plt.xticks([1,2,3,4], x_label)
    plt.savefig(fig_save_path+"/classe_"+c+".png")
    plt.close()
    
#%% PATH INIZIALIZATION
original_dataset_path = "D:/DATASET_Tesi_marzo2024/"
list_of_dataset = os.listdir(original_dataset_path)

LS_2c_path_ds = original_dataset_path + list_of_dataset[0] +"/DATASET_2classes/"
LS_3c_path_ds = original_dataset_path + list_of_dataset[0] +"/DATASET_3classes/"
RG_3c_path_ds = original_dataset_path + list_of_dataset[1] +"/DATASET/"
RT_3c_path_ds = original_dataset_path + list_of_dataset[2] +"/DATASET/"

LS_2c_path_ks = original_dataset_path + list_of_dataset[0] +"/k-net+swin/TEST_2classes/"
LS_3c_path_ks = original_dataset_path + list_of_dataset[0] +"/k-net+swin/TEST_3classes/"
RG_3c_path_ks = original_dataset_path + list_of_dataset[1] +"/k-net+swin/TEST/"
RT_3c_path_ks = original_dataset_path + list_of_dataset[2] +"/k-net+swin/TEST/"

dataset_for_loop = [LS_2c_path_ds,LS_3c_path_ds,RG_3c_path_ds,RT_3c_path_ds]

LS_3c_dict = {
    'path_ds': LS_3c_path_ds,
    'path_ks': LS_3c_path_ks,
    'path_GT': "/manual/",
    'C1_name': 'Class 1',
    'C2_name': 'Class 2'
    }

RG_dict = {
    'path_ds': original_dataset_path + list_of_dataset[1] +"/DATASET/",
    'path_ks': original_dataset_path + list_of_dataset[1] +"/k-net+swin/TEST/",
    'path_GT': "/manual/",
    'C1_name': 'Healthy',
    'C2_name': 'Sclerotic'
    }

RT_dict = {
    'path_ds': original_dataset_path + list_of_dataset[2] +"/DATASET/",
    'path_ks': original_dataset_path + list_of_dataset[2] +"/k-net+swin/TEST/",
    'path_GT': "/manual/",
    'C1_name': 'Healthy',
    'C2_name': 'Atro'
    }

dataset_dict = {
    # 'Liver HE steatosis 2c': LS_2c_dict,
    # 'Liver HE steatosis 3c': LS_3c_dict,
    # 'Renal PAS glomeruli': RG_dict,
    'Renal PAS tubuli': RT_dict
    }

#%% FOR CYCLE
for dataset in dataset_dict:
    for subset in ['val','test']:
        image_list = os.listdir(dataset_dict[dataset]['path_ds']+subset+"/"+dataset_dict[dataset]['path_GT'])
        metrics_matrix = []
        image_counter = 0
        metrics_save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V2/"+dataset+"/"+subset+"/"
        for name in tqdm(image_list):
            # Inizializzazione path e immagini importanti
            original_image_path = dataset_dict[dataset]['path_ds']+subset+"/image/" + name
            GT_mask_path = dataset_dict[dataset]['path_ds']+subset+"/manual/" + name
            SI_mask_path = dataset_dict[dataset]['path_ks']+"RESULTS/"+subset+"/mask/"+name
            
            PERT_path_20_seg = dataset_dict[dataset]['path_ks']+'/RESULTS_perturbation/'+subset+'/mask/' + name + "/"
            PERT_path_20_softmax = dataset_dict[dataset]['path_ks']+'/RESULTS_perturbation/'+subset+'/softmax/' + name + "/"
            
            MC_path_20_seg = dataset_dict[dataset]['path_ks']+'/RESULTS_MC/'+subset+'/mask/' + name + "/"
            MC_path_20_softmax = dataset_dict[dataset]['path_ks']+'/RESULTS_MC/'+subset+'/softmax/' + name + "/"
            
            save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V2/"+dataset+"/"+subset+"/"+name+"/"
            if not os.path.isdir(save_path): os.makedirs(save_path)
            
            GT_mask_orig = np.array(Img.open(GT_mask_path))/255
            SI_mask_orig = np.array(Img.open(SI_mask_path))/255

            GT_mask_C1,GT_mask_C2 = mask_splitter(GT_mask_orig)
            SI_mask_C1,SI_mask_C2 = mask_splitter(SI_mask_orig)
            
            DIM = np.shape(GT_mask_orig)
            N = 20
            c = 3
            
            
            # Generazione seg_matrix
            seg_matrix_PERT = seg_matrix_gen(PERT_path_20_seg, DIM, N)
            seg_matrix_MC = seg_matrix_gen(MC_path_20_seg, DIM, N)
            
            softmax_matrix_PERT = softmax_matrix_gen(PERT_path_20_softmax, DIM, c ,N)
            softmax_matrix_MC = softmax_matrix_gen(MC_path_20_softmax, DIM, c ,N)
            
            #creo binary entropy map
            bin_ent_map_c1_PERT = binary_entropy_map(softmax_matrix_PERT[:,:,0,:])
            bin_ent_map_c2_PERT = binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
            bin_ent_map_c1_MC = binary_entropy_map(softmax_matrix_MC[:,:,0,:])
            bin_ent_map_c2_MC = binary_entropy_map(softmax_matrix_MC[:,:,1,:])            
            
            softmax_matrix = softmax_matrix_MC
            uncert_map = bin_ent_map_c1_MC
            SI_mask = SI_mask_C1
            
            mean_softmax = np.mean(softmax_matrix,axis=-1)    
            mask_avg = np.argmax(mean_softmax,axis=-1)
            
            
            dice_c1_MC = []
            dice_c2_MC = []
            dice_c1_PERT = []
            dice_c2_PERT = []
            precision_c1_MC = []
            precision_c2_MC = []
            precision_c1_PERT = []
            precision_c2_PERT = []
            recall_c1_MC = []
            recall_c2_MC = []
            recall_c1_PERT = []
            recall_c2_PERT = []
            
            Th_array = np.arange(0,1,0.01)
            for Th in Th_array:
                mask_auto_c1_MC = mask_generator_for_calcoli(SI_mask_C1,bin_ent_map_c1_MC,Th,softmax_matrix_MC)
                mask_auto_c2_MC = mask_generator_for_calcoli(SI_mask_C2,bin_ent_map_c2_MC,Th,softmax_matrix_MC)
                mask_auto_c1_PERT = mask_generator_for_calcoli(SI_mask_C1,bin_ent_map_c1_PERT,Th,softmax_matrix_PERT)
                mask_auto_c2_PERT = mask_generator_for_calcoli(SI_mask_C2,bin_ent_map_c2_PERT,Th,softmax_matrix_PERT)
                
                dice_c1_MC.append(dice(mask_auto_c1_MC,GT_mask_C1))
                dice_c2_MC.append(dice(mask_auto_c2_MC,GT_mask_C2))
                dice_c1_PERT.append(dice(mask_auto_c1_PERT,GT_mask_C1))
                dice_c2_PERT.append(dice(mask_auto_c2_PERT,GT_mask_C2))
                
                precision_c1_MC.append(precision(mask_auto_c1_MC,GT_mask_C1))
                precision_c2_MC.append(precision(mask_auto_c2_MC,GT_mask_C2))
                precision_c1_PERT.append(precision(mask_auto_c1_PERT,GT_mask_C1))
                precision_c2_PERT.append(precision(mask_auto_c2_PERT,GT_mask_C2))
                
                recall_c1_MC.append(recall(mask_auto_c1_MC,GT_mask_C1))
                recall_c2_MC.append(recall(mask_auto_c2_MC,GT_mask_C2))
                recall_c1_PERT.append(recall(mask_auto_c1_PERT,GT_mask_C1))
                recall_c2_PERT.append(recall(mask_auto_c2_PERT,GT_mask_C2))
            
            plt.figure(figsize=(20,20))
            plt.plot(Th_array,dice_c1_MC)
            plt.plot(Th_array,dice_c2_MC)
            plt.plot(Th_array,dice_c1_PERT)
            plt.plot(Th_array,dice_c2_PERT)
            plt.axhline(y = dice(SI_mask_C1,GT_mask_C1), color = 'r', linestyle = '-')
            plt.xlim(0,1)
            plt.ylim(top=1)
            plt.legend(['c1 MC','c2 MC', 'c1 PERT', 'c2 PERT'])
            plt.savefig(save_path + "dice.png")
            plt.close()
            
            plt.figure(figsize=(20,20))
            plt.plot(Th_array,precision_c1_MC)
            plt.plot(Th_array,precision_c2_MC)
            plt.plot(Th_array,precision_c1_PERT)
            plt.plot(Th_array,precision_c2_PERT)
            plt.axhline(y = precision(SI_mask_C1,GT_mask_C1), color = 'r', linestyle = '-')
            plt.xlim(0,1)
            plt.ylim(top=1)
            plt.legend(['c1 MC','c2 MC', 'c1 PERT', 'c2 PERT'])
            plt.savefig(save_path + "precision.png")
            plt.close()
            
            plt.figure(figsize=(20,20))
            plt.plot(Th_array,recall_c1_MC)
            plt.plot(Th_array,recall_c2_MC)
            plt.plot(Th_array,recall_c1_PERT)
            plt.plot(Th_array,recall_c2_PERT)
            plt.axhline(y = recall(SI_mask_C1,GT_mask_C1), color = 'r', linestyle = '-')
            plt.xlim(0,1)
            plt.ylim(top=1)
            plt.legend(['c1 MC','c2 MC', 'c1 PERT', 'c2 PERT'])
            plt.savefig(save_path + "recall.png")
            plt.close()   