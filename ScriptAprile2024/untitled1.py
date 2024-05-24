# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:33:42 2024

@author: willy
"""

# %% IMPORT
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Img
from tqdm import tqdm

# %% PRIMARY FUNCTIONS


def dice(mask_automatic, mask_manual):
    TP_mask = np.multiply(mask_automatic, mask_manual)
    TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(
        int), TP_mask.astype(int)).astype(bool)
    FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(
        int), TP_mask.astype(int)).astype(bool)
    FN = FN_mask.sum()

    if TP == 0 and FN == 0 and FP == 0:
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

def mask_generator_for_calcoli(SI_mask_C1,SI_mask_C2,uncert_map_C1,uncert_map_C2,Th,softmax_matrix): #da fare una volta per classe
    
    mean_softmax = np.mean(softmax_matrix,axis=-1)    
    mask_avg = np.argmax(mean_softmax,axis=-1)/2
    
    mask_avg_C1,mask_avg_C2 = mask_splitter(mask_avg)
    mask_avg_C1=mask_avg_C1.astype(bool)
    mask_avg_C2=mask_avg_C2.astype(bool)
    
    mask_uncert_C1 = uncert_map_C1 > Th   #(0.1, 0.2, 0.3, ...)
    mask_cert_C1 = (~mask_uncert_C1) & mask_avg_C1
    mask_uncert_C2 = uncert_map_C2 > Th   #(0.1, 0.2, 0.3, ...)
    mask_cert_C2 = (~mask_uncert_C2) & mask_avg_C2
    
    mask_FP_C1 = np.copy(mask_uncert_C1)
    mask_FN_C1 = np.copy(mask_cert_C1)
    mask_FP_C2 = np.copy(mask_uncert_C2)
    mask_FN_C2 = np.copy(mask_cert_C2)
    
    mask_auto_C1 = np.copy(SI_mask_C1)
    mask_auto_C2 = np.copy(SI_mask_C2)
    
    mask_auto_C1[mask_FP_C1] = False
    mask_auto_C1[mask_FN_C1] = True
    mask_auto_C2[mask_FP_C2] = False
    mask_auto_C2[mask_FN_C2] = True
    
    # mask_auto = mask_auto_C1*0.5+mask_auto_C2
    
    return mask_auto_C1,mask_auto_C2

def mask_splitter(mask):
    mask_C1 = np.copy(mask)
    mask_C2 = np.copy(mask)

    mask_C1[mask > 0.8] = 0
    mask_C1[mask_C1 > 0] = 1
    mask_C2[mask < 0.8] = 0
    mask_C2[mask_C2 > 0] = 1
    return mask_C1, mask_C2

# %% MEGAMATRIX GENERATION
# Softmax matrix generator


def softmax_matrix_gen(softmax_path, DIM, c, N):
    softmax_matrix = np.zeros((DIM[0], DIM[1], c, N), dtype=np.float32)
    counter = 0
    for num in os.listdir(softmax_path):
        for n_class in range(c):
            st0 = np.float32(
                (np.load(softmax_path + "/" + num)['softmax'])[:, :, n_class])
            softmax_matrix[:, :, n_class, counter] = np.copy(st0)
        counter += 1
    return softmax_matrix

# Seg Matrix generator


def seg_matrix_gen(mask_path, DIM, N):
    seg_matrix = np.zeros((DIM[0], DIM[1], N))
    counter = 0
    for num in os.listdir(mask_path):
        temp = Img.open(mask_path + num)
        seg_matrix[:, :, counter] = np.array(temp)/255
        counter += 1
    values = np.unique(seg_matrix)
    classes = len(values)
    standard_values = [0, 0.5, 1]
    for i in range(classes):
        seg_matrix[seg_matrix == values[i]] = standard_values[i]
    if np.max(seg_matrix) == 0.5:
        seg_matrix[seg_matrix == 0.5] = 1
    return seg_matrix

# %% Map generator functions

# Binary Entropy Map


def binary_entropy_map(softmax_matrix):
    p_mean = np.mean(softmax_matrix, axis=2)
    p_mean[p_mean == 0] = 1e-8
    p_mean[p_mean == 1] = 1-1e-8
    HB_pert = -(np.multiply(p_mean, np.log2(p_mean)) +
                np.multiply((1-p_mean), np.log2(1-p_mean)))
    HB_pert[np.isnan(HB_pert)] = np.nanmin(HB_pert)
    return HB_pert


# Shannon Entropy Map
def shannon_entropy_map(softmax_matrix):
    shape = np.shape(softmax_matrix)
    shannon_entropy_map = np.zeros([shape[0], shape[1]]).astype(float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ij_vector = softmax_matrix[i, j, :]
            shannon_entropy_map[i, j] = -(np.multiply(
                ij_vector[ij_vector != 0], np.log2(ij_vector[ij_vector != 0]))).sum()
    return shannon_entropy_map

# Std Map


def std_map(softmax_matrix): return np.nanstd(softmax_matrix, axis=2)

# Dice Mat


def dice_mat_map(seg_matrix, N):
    dice_mat = -np.ones((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dice_mat[i, j] = dice(seg_matrix[:, :, i], seg_matrix[:, :, j])
    dice_mat[dice_mat < 0] = np.nan
    return dice_mat

# %% PLOT GENERATOR


def boxplot_generator(dice_GTSI, dice_MC, dice_PERT, TOT_dice, fig_save_path, name, dataset, subset, classe):
    data = [dice_GTSI, dice_MC, dice_PERT, TOT_dice]

    plt.figure(figsize=(10, 7))
    x_label = ['Single Inference Dice', 'Monte Carlo Dices',
               'Perturbations Dices', 'Combined Dices']
    plt.boxplot(data)
    plt.title("Image " + name[:-4] + " of " + dataset +
              " (subset " + subset + ") - class " + str(classe))
    plt.ylabel('DICE')
    plt.xticks([1, 2, 3, 4], x_label)
    plt.savefig(fig_save_path)
    plt.close()

def unc_maps_plots(
        dataset,
        subset,
        name,
        original_image,
        GT_mask,
        SI_mask,
        bin_ent_map1,
        sha_ent_map1,
        std_map1,
        dice_mat1,
        bin_ent_map2,
        sha_ent_map2,
        std_map2,
        dice_mat2,
        save_path):

    plt.figure(figsize=(10, 30))
    plt.suptitle('IMAGE: '+name[:-4]+", DATASET: "+dataset+", SUBSET: "+subset)
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(132)
    plt.title("Ground Truth Mask")
    plt.imshow(GT_mask)
    plt.subplot(133)
    plt.title("Single Inference Mask")
    plt.imshow(SI_mask)
    plt.savefig(save_path + "Image_GT_SI.png")
    plt.close()

    plt.figure(figsize=(15, 60))
    plt.suptitle('IMAGE: '+name[:-4]+", DATASET: "+dataset+", SUBSET: "+subset)

    plt.subplot(141)
    plt.title("Binary Entropy Map")
    plt.imshow(bin_ent_map1)
    plt.colorbar()

    plt.subplot(142)
    plt.title("Shannon Entropy Map")
    plt.imshow(sha_ent_map1)
    plt.colorbar()

    plt.subplot(143)
    plt.title("Std Map")
    plt.imshow(std_map1)
    plt.colorbar()

    plt.subplot(144)
    plt.title("Inter-dice Matrix")
    plt.imshow(dice_mat1)
    plt.colorbar()

    plt.savefig(save_path + "uncertainty_maps_C1.png")
    plt.close()

    plt.figure(figsize=(15, 60))
    plt.suptitle('IMAGE: '+name[:-4]+", DATASET: "+dataset+", SUBSET: "+subset)

    plt.subplot(141)
    plt.title("Binary Entropy Map")
    plt.imshow(bin_ent_map2)
    plt.colorbar()

    plt.subplot(142)
    plt.title("Shannon Entropy Map")
    plt.imshow(sha_ent_map2)
    plt.colorbar()

    plt.subplot(143)
    plt.title("Std Map")
    plt.imshow(std_map2)
    plt.colorbar()

    plt.subplot(144)
    plt.title("Inter-dice Matrix")
    plt.imshow(dice_mat2)
    plt.colorbar()

    plt.savefig(save_path + "uncertainty_maps_C2.png")
    plt.close()


# %% PATH INIZIALIZATION
original_dataset_path = "D:/DATASET_Tesi_marzo2024/"
list_of_dataset = os.listdir(original_dataset_path)

LS_3c_dict = {
    'path_ds': original_dataset_path + list_of_dataset[0] + "/DATASET_3classes/",
    'path_ks': original_dataset_path + list_of_dataset[0] + "/k-net+swin/TEST_3classes/",
    'path_GT': "/manual/",
    'C1_name': 'Class 1',
    'C2_name': 'Class 2'
}

RG_dict = {
    'path_ds': original_dataset_path + list_of_dataset[1] + "/DATASET/",
    'path_ks': original_dataset_path + list_of_dataset[1] + "/k-net+swin/TEST/",
    'path_GT': "/manual/",
    'C1_name': 'Healthy',
    'C2_name': 'Sclerotic'
}

RT_dict = {
    'path_ds': original_dataset_path + list_of_dataset[2] + "/DATASET/",
    'path_ks': original_dataset_path + list_of_dataset[2] + "/k-net+swin/TEST/",
    'path_GT': "/manual/",
    'C1_name': 'Healthy',
    'C2_name': 'Atro'
}

dataset_dict = {
    # 'Liver HE steatosis 2c': LS_2c_dict,
    'Liver HE steatosis 3c': LS_3c_dict,
    'Renal PAS glomeruli': RG_dict,
    'Renal PAS tubuli': RT_dict
}

# %% P R O V A
# dataset = 'Renal PAS tubuli'
# subset = 'test'
# name = '1004761_2.png'

for dataset in tqdm(dataset_dict):
    for subset in tqdm(['val', 'test']):
        if subset == 'test': continue
        image_list = os.listdir(
            dataset_dict[dataset]['path_ds']+subset+"/"+dataset_dict[dataset]['path_GT'])
        subset_save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V3/"+dataset+"/"+subset+"/"
        if not os.path.isdir(subset_save_path):
            os.makedirs(subset_save_path)

        mean_MC_dice_C1 = []
        mean_PERT_dice_C1 = []
        mean_MC_dice_C2 = []
        mean_PERT_dice_C2 = []
        SI_dice_array_C1 = []
        SI_dice_array_C2 = []
        
        bin_sum_array_MC_C1 = []
        sha_sum_array_MC_C1 = []
        std_sum_array_MC_C1 = []
        dmat_mean_array_MC_C1 = []
        dmat_std_array_MC_C1 = []
        
        bin_sum_array_MC_C2 = []
        sha_sum_array_MC_C2 = []
        std_sum_array_MC_C2 = []
        dmat_mean_array_MC_C2 = []
        dmat_std_array_MC_C2 = []
        
        bin_sum_array_PERT_C1 = []
        sha_sum_array_PERT_C1 = []
        std_sum_array_PERT_C1 = []
        dmat_mean_array_PERT_C1 = []
        dmat_std_array_PERT_C1 = []
        
        bin_sum_array_PERT_C2 = []
        sha_sum_array_PERT_C2 = []
        std_sum_array_PERT_C2 = []
        dmat_mean_array_PERT_C2 = []
        dmat_std_array_PERT_C2 = []
        
        dice_GTSI_C1_array = []
        dice_GTSI_C2_array = []
        
        th_max_dice_per_image_MC_C1 = []
        th_max_dice_per_image_MC_C2 = []
        th_max_dice_per_image_PERT_C1 = []
        th_max_dice_per_image_PERT_C2 = []
        
        th_max_precision_per_image_MC_C1 = []
        th_max_precision_per_image_MC_C2 = []
        th_max_precision_per_image_PERT_C1 = []
        th_max_precision_per_image_PERT_C2 = []
        
        th_max_recall_per_image_MC_C1 = []
        th_max_recall_per_image_MC_C2 = []
        th_max_recall_per_image_PERT_C1 = []
        th_max_recall_per_image_PERT_C2 = []

        count_image = 0
        for name in tqdm(image_list):
            # %% Inizializzazione path e immagini importanti
            original_image_path = dataset_dict[dataset]['path_ds'] + \
                subset+"/image/" + name
            GT_mask_path = dataset_dict[dataset]['path_ds'] + \
                subset+"/manual/" + name
            SI_mask_path = dataset_dict[dataset]['path_ks'] + \
                "RESULTS/"+subset+"/mask/"+name

            PERT_path_20_seg = dataset_dict[dataset]['path_ks'] + \
                '/RESULTS_perturbation/'+subset+'/mask/' + name + "/"
            PERT_path_20_softmax = dataset_dict[dataset]['path_ks'] + \
                '/RESULTS_perturbation/'+subset+'/softmax/' + name + "/"

            MC_path_20_seg = dataset_dict[dataset]['path_ks'] + \
                '/RESULTS_MC/'+subset+'/mask/' + name + "/"
            MC_path_20_softmax = dataset_dict[dataset]['path_ks'] + \
                '/RESULTS_MC/'+subset+'/softmax/' + name + "/"

            save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V3/"+dataset+"/"+subset+"/"+name+"/"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            original_image = np.array(Img.open(original_image_path))

            GT_mask_orig = np.array(Img.open(GT_mask_path))/255
            SI_mask_orig = np.array(Img.open(SI_mask_path))/255

            GT_mask_C1, GT_mask_C2 = mask_splitter(GT_mask_orig)
            SI_mask_C1, SI_mask_C2 = mask_splitter(SI_mask_orig)

            DIM = np.shape(GT_mask_orig)
            N = 20
            c = 3

# %% SOFTMAX MATRIX GENERATOR

            softmax_matrix_PERT = softmax_matrix_gen(
                PERT_path_20_softmax, DIM, c, N)
            softmax_matrix_MC = softmax_matrix_gen(
                MC_path_20_softmax, DIM, c, N)
            
            #creo binary entropy map
            bin_ent_map_c1_PERT = binary_entropy_map(softmax_matrix_PERT[:,:,0,:])
            bin_ent_map_c2_PERT = binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
            bin_ent_map_c1_MC = binary_entropy_map(softmax_matrix_MC[:,:,0,:])
            bin_ent_map_c2_MC = binary_entropy_map(softmax_matrix_MC[:,:,1,:])            
                                  
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
                mask_auto_c1_MC,mask_auto_c2_MC = mask_generator_for_calcoli(SI_mask_C1,SI_mask_C2,bin_ent_map_c1_MC,bin_ent_map_c2_MC,Th,softmax_matrix_MC)
                mask_auto_c1_PERT,mask_auto_c2_PERT = mask_generator_for_calcoli(SI_mask_C1,SI_mask_C2,bin_ent_map_c1_PERT,bin_ent_map_c2_PERT,Th,softmax_matrix_PERT)
                
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
                
            dice_c1_MC = np.array(dice_c1_MC)
            dice_c1_MC[np.isnan(dice_c1_MC)]=0 
            dice_c2_MC = np.array(dice_c2_MC)
            dice_c2_MC[np.isnan(dice_c2_MC)]=0
            dice_c1_PERT = np.array(dice_c1_PERT)
            dice_c1_PERT[np.isnan(dice_c1_PERT)]=0
            dice_c2_PERT = np.array(dice_c2_PERT)
            dice_c2_PERT[np.isnan(dice_c2_PERT)]=0
            
            precision_c1_MC = np.array(precision_c1_MC)
            precision_c1_MC[np.isnan(precision_c1_MC)]=0 
            precision_c2_MC = np.array(precision_c2_MC)
            precision_c2_MC[np.isnan(precision_c2_MC)]=0
            precision_c1_PERT = np.array(precision_c1_PERT)
            precision_c1_PERT[np.isnan(precision_c1_PERT)]=0
            precision_c2_PERT = np.array(precision_c2_PERT)
            precision_c2_PERT[np.isnan(precision_c2_PERT)]=0
            
            recall_c1_MC = np.array(recall_c1_MC)
            recall_c1_MC[np.isnan(recall_c1_MC)]=0 
            recall_c2_MC = np.array(recall_c2_MC)
            recall_c2_MC[np.isnan(recall_c2_MC)]=0
            recall_c1_PERT = np.array(recall_c1_PERT)
            recall_c1_PERT[np.isnan(recall_c1_PERT)]=0
            recall_c2_PERT = np.array(recall_c2_PERT)
            recall_c2_PERT[np.isnan(recall_c2_PERT)]=0
            
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
            
            th_max_dice_per_image_MC_C1.append(Th_array[np.argmax(dice_c1_MC)])
            th_max_dice_per_image_MC_C2.append(Th_array[np.argmax(dice_c2_MC)])
            th_max_dice_per_image_PERT_C1.append(Th_array[np.argmax(dice_c1_PERT)])
            th_max_dice_per_image_PERT_C2.append(Th_array[np.argmax(dice_c2_PERT)])
            
            th_max_precision_per_image_MC_C1.append(Th_array[np.argmax(precision_c1_MC)])
            th_max_precision_per_image_MC_C2.append(Th_array[np.argmax(precision_c2_MC)])
            th_max_precision_per_image_PERT_C1.append(Th_array[np.argmax(precision_c1_PERT)])
            th_max_precision_per_image_PERT_C2.append(Th_array[np.argmax(precision_c2_PERT)])
            
            th_max_recall_per_image_MC_C1.append(Th_array[np.argmax(recall_c1_MC)])
            th_max_recall_per_image_MC_C2.append(Th_array[np.argmax(recall_c2_MC)])
            th_max_recall_per_image_PERT_C1.append(Th_array[np.argmax(recall_c1_PERT)])
            th_max_recall_per_image_PERT_C2.append(Th_array[np.argmax(recall_c2_PERT)])
            
        plt.close()
        plt.figure(figsize=(30,12))
        plt.suptitle("Threshold with max dice, precision, recall per image - MC C1")
        plt.subplot(131)
        plt.plot(image_list,th_max_dice_per_image_MC_C1)
        plt.title("Dice")
        plt.subplot(132)
        plt.plot(image_list,th_max_precision_per_image_MC_C1)
        plt.title("Precision")
        plt.subplot(133)
        plt.plot(image_list,th_max_recall_per_image_MC_C1)
        plt.title("Recall")
        plt.savefig(subset_save_path+"_threshold_per_image_MC_C1.png")
        plt.close()
        
        plt.close()
        plt.figure(figsize=(30,12))
        plt.suptitle("Threshold with max dice, precision, recall per image - MC C2")
        plt.subplot(131)
        plt.plot(image_list,th_max_dice_per_image_MC_C2)
        plt.title("Dice")
        plt.subplot(132)
        plt.plot(image_list,th_max_precision_per_image_MC_C2)
        plt.title("Precision")
        plt.subplot(133)
        plt.plot(image_list,th_max_recall_per_image_MC_C2)
        plt.title("Recall")
        plt.savefig(subset_save_path+"threshold_per_image_MC_C2.png")
        plt.close()
        
        plt.close()
        plt.figure(figsize=(30,12))
        plt.suptitle("Threshold with max dice, precision, recall per image - PERT C1")
        plt.subplot(131)
        plt.plot(image_list,th_max_dice_per_image_PERT_C1)
        plt.title("Dice")
        plt.subplot(132)
        plt.plot(image_list,th_max_precision_per_image_PERT_C1)
        plt.title("Precision")
        plt.subplot(133)
        plt.plot(image_list,th_max_recall_per_image_PERT_C1)
        plt.title("Recall")
        plt.savefig(subset_save_path+"threshold_per_image_PERT_C1.png")
        plt.close()
        
        plt.close()
        plt.figure(figsize=(30,12))
        plt.suptitle("Threshold with max dice, precision, recall per image - PERT C2")
        plt.subplot(131)
        plt.plot(image_list,th_max_dice_per_image_PERT_C2)
        plt.title("Dice")
        plt.subplot(132)
        plt.plot(th_max_precision_per_image_PERT_C2)
        plt.title("Precision")
        plt.subplot(133)
        plt.plot(image_list,th_max_recall_per_image_PERT_C2)
        plt.title("Recall")
        plt.savefig(subset_save_path+"threshold_per_image_PERT_C2.png")
        plt.close()
        
        dict_thresholds = {
            'Image': image_list,
            'MC_C1_dice': th_max_dice_per_image_MC_C1,
            'MC_C2_dice': th_max_dice_per_image_MC_C2,
            'PERT_C1_dice': th_max_dice_per_image_PERT_C1,
            'PERT_C2_dice': th_max_dice_per_image_PERT_C2,
            'MC_C1_precision': th_max_precision_per_image_MC_C1,
            'MC_C2_precision': th_max_precision_per_image_MC_C2,
            'PERT_C1_precision': th_max_precision_per_image_PERT_C1,
            'PERT_C2_precision': th_max_precision_per_image_PERT_C2,
            'MC_C1_recall': th_max_recall_per_image_MC_C1,
            'MC_C2_recall': th_max_recall_per_image_MC_C2,
            'PERT_C1_recall': th_max_recall_per_image_PERT_C1,
            'PERT_C2_recall': th_max_recall_per_image_PERT_C2
            }
        
        df_th = pd.DataFrame(dict_thresholds)
        df_th.to_csv(subset_save_path+"Max_Thresholds_per_image.csv",index=False)
        
        