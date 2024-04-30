# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:50:34 2024

@author: willy
"""

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

def mask_splitter(mask):
    mask_C1 = np.copy(mask)
    mask_C2 = np.copy(mask)

    mask_C1[mask>0.8]=0
    mask_C1[mask>0]=1
    mask_C2[mask<0.8]=0
    # mask_C2[mask>0]=1
    return mask_C1, mask_C2

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

#%% PLOT GENERATOR
def boxplot_generator(dice_GTSI, dice_MC, dice_PERT, TOT_dice, fig_save_path, name, dataset, subset, classe):
    data = [dice_GTSI, dice_MC, dice_PERT, TOT_dice]
    
    plt.figure(figsize =(10, 7))
    x_label = ['Single Inference Dice', 'Monte Carlo Dices', 'Perturbations Dices', 'Combined Dices']
    plt.boxplot(data)
    plt.title("Image " + name[:-4] + " of " + dataset + " (subset " + subset + ") - class " + str(classe))
    plt.ylabel('DICE')
    plt.xticks([1,2,3,4], x_label)
    plt.savefig(fig_save_path)
    plt.close()
    
def unc_maps_plots(
        dataset,
        subset,
        name,
        classe,
        original_image,
        GT_mask,
        bin_ent_map1,
        sha_ent_map1,
        std_map1,
        dice_mat1,
        bin_ent_map2,
        sha_ent_map2,
        std_map2,
        dice_mat2,
        save_path
        ):
    
    plt.figure(figsize=(25,60))
    plt.suptitle('IMAGE: '+name[:-4]+" DATASET: "+dataset+" SUBSET: "+subset+" - classe "+str(classe))
    
    plt.subplot(251)
    plt.title("Original Image")
    plt.imshow(original_image)
    
    plt.subplot(252)
    plt.title("Binary Entropy Map")
    plt.imshow(bin_ent_map1)
    plt.colorbar()
    
    plt.subplot(253)
    plt.title("Shannon Entropy Map")
    plt.imshow(sha_ent_map1)
    plt.colorbar()
    
    plt.subplot(254)
    plt.title("Std Map")
    plt.imshow(std_map1)
    plt.colorbar()
    
    plt.subplot(255)
    plt.title("Inter-dice Matrix")
    plt.imshow(dice_mat1)
    plt.colorbar()
    
    plt.subplot(256)
    plt.title("Ground Truth Mask")
    plt.imshow(original_image)
    
    plt.subplot(257)
    plt.title("Binary Entropy Map")
    plt.imshow(bin_ent_map2)
    plt.colorbar()
    
    plt.subplot(258)
    plt.title("Shannon Entropy Map")
    plt.imshow(sha_ent_map2)
    plt.colorbar()
    
    plt.subplot(259)
    plt.title("Std Map")
    plt.imshow(std_map2)
    plt.colorbar()
    
    plt.subplot(2,5,10)
    plt.title("Inter-dice Matrix")
    plt.imshow(dice_mat2)
    plt.colorbar()
    
    plt.savefig(save_path+"unc_maps_plots.png")
    plt.close()

#%% PATH INIZIALIZATION
original_dataset_path = "D:/DATASET_Tesi_marzo2024/"
list_of_dataset = os.listdir(original_dataset_path)

LS_3c_dict = {
    'path_ds': original_dataset_path + list_of_dataset[0] +"/DATASET_3classes/",
    'path_ks': original_dataset_path + list_of_dataset[0] +"/k-net+swin/TEST_3classes/",
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

#%% P R O V A
# dataset = 'Renal PAS tubuli'
# subset = 'test'
# name = '1004761_2.png'

for dataset in tqdm(dataset_dict):
    for subset in tqdm(['val','test']):
        image_list = os.listdir(dataset_dict[dataset]['path_ds']+subset+"/"+dataset_dict[dataset]['path_GT'])
        for name in tqdm(image_list):
            #%% Inizializzazione path e immagini importanti
            original_image_path = dataset_dict[dataset]['path_ds']+subset+"/image/" + name
            GT_mask_path = dataset_dict[dataset]['path_ds']+subset+"/manual/" + name
            SI_mask_path = dataset_dict[dataset]['path_ks']+"RESULTS/"+subset+"/mask/"+name
            
            PERT_path_20_seg = dataset_dict[dataset]['path_ks']+'/RESULTS_perturbation/'+subset+'/mask/' + name + "/"
            PERT_path_20_softmax = dataset_dict[dataset]['path_ks']+'/RESULTS_perturbation/'+subset+'/softmax/' + name + "/"
            
            MC_path_20_seg = dataset_dict[dataset]['path_ks']+'/RESULTS_MC/'+subset+'/mask/' + name + "/"
            MC_path_20_softmax = dataset_dict[dataset]['path_ks']+'/RESULTS_MC/'+subset+'/softmax/' + name + "/"
            
            save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V2/"+dataset+"/"+subset+"/"+name+"/"
            if not os.path.isdir(save_path): os.makedirs(save_path)
            
            original_image = np.array(Img.open(original_image_path))
            
            GT_mask_orig = np.array(Img.open(GT_mask_path))/255
            SI_mask_orig = np.array(Img.open(SI_mask_path))/255
            
            GT_mask_C1,GT_mask_C2 = mask_splitter(GT_mask_orig)
            SI_mask_C1,SI_mask_C2 = mask_splitter(SI_mask_orig)
            
            DIM = np.shape(GT_mask_orig)
            N = 20
            c = 3
            
            #%% BOXPLOT
            fig_save_path = save_path + "plots/"
            if not os.path.isdir(fig_save_path): os.makedirs(fig_save_path)
            
            dice_GTSI_C1 = dice(GT_mask_C1,SI_mask_C1)
            dice_MC_C1 = []
            dice_PERT_C1 = []
            TOT_dice_C1 = []
            
            dice_GTSI_C2 = dice(GT_mask_C2,SI_mask_C2)
            dice_MC_C2 = []
            dice_PERT_C2 = []
            TOT_dice_C2 = []
            
            for n_sample in os.listdir(MC_path_20_seg):
                actual_seg_MC = np.array(Img.open(MC_path_20_seg+n_sample))/255
                seg_MC_C1,seg_MC_C2 = mask_splitter(actual_seg_MC)
                actual_seg_PERT = np.array(Img.open(PERT_path_20_seg+n_sample))/255
                seg_PERT_C1,seg_PERT_C2 = mask_splitter(actual_seg_PERT)
                
                dice_MC1 = dice(GT_mask_C1,seg_MC_C1)
                dice_PERT1 = dice(GT_mask_C1,seg_PERT_C1)
                
                dice_MC_C1.append(dice_MC1)
                dice_PERT_C1.append(dice_PERT1)
                TOT_dice_C1.append(dice_MC1)
                TOT_dice_C1.append(dice_PERT1)
                
                dice_MC2 = dice(GT_mask_C2,seg_MC_C2)
                dice_PERT2 = dice(GT_mask_C2,seg_PERT_C2)
                
                dice_MC_C2.append(dice_MC2)
                dice_PERT_C2.append(dice_PERT2)
                TOT_dice_C2.append(dice_MC2)
                TOT_dice_C2.append(dice_PERT2)
            
            dice_MC_C1 = np.array(dice_MC_C1)
            dice_PERT_C1 = np.array(dice_PERT_C1)
            TOT_dice_C1 = np.array(TOT_dice_C1)
            
            dice_MC_C2 = np.array(dice_MC_C2)
            dice_PERT_C2 = np.array(dice_PERT_C2)
            TOT_dice_C2 = np.array(TOT_dice_C2)
            
            boxplot_generator(dice_GTSI_C1,dice_MC_C1,dice_PERT_C1,TOT_dice_C1,fig_save_path+"boxplot_MC_C1.png",name,dataset,subset,1)
            boxplot_generator(dice_GTSI_C2,dice_MC_C2,dice_PERT_C2,TOT_dice_C2,fig_save_path+"boxplot_MC_C2.png",name,dataset,subset,2)

#%%
            # unc_maps_plots(dataset, subset, name, classe, original_image, GT_mask_orig, bin_ent_map1, sha_ent_map1, std_map1, dice_mat1, bin_ent_map2, sha_ent_map2, std_map2, dice_mat2, save_path)
    


