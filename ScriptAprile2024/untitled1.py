# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:24:42 2024

@author: willy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Img
from tqdm import tqdm
from skimage import measure
import copy

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

def mask_avg_gen(softmax_matrix):
    temp = np.argmax(np.mean(softmax_matrix,axis=-1),axis=-1)
    if temp.max() > 1: 
        mask_avg = temp/2
    else:
        mask_avg = temp
    return mask_avg
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

LS_2c_dict = {
    'path_ds': original_dataset_path + list_of_dataset[0] + "/DATASET_2classes/",
    'path_ks': original_dataset_path + list_of_dataset[0] + "/k-net+swin/TEST_2classes/",
    'path_GT': "/manual/"
}

LS_3c_dict = {
    'path_ds': original_dataset_path + list_of_dataset[0] + "/DATASET_3classes/",
    'path_ks': original_dataset_path + list_of_dataset[0] + "/k-net+swin/TEST_3classes/",
    'path_GT': "/manual/",
    'C1_name': 'Class 1',
    'C2_name': 'Class 2'
}

# RG_dict = {
#     'path_ds': original_dataset_path + list_of_dataset[1] + "/DATASET/",
#     'path_ks': original_dataset_path + list_of_dataset[1] + "/k-net+swin/TEST/",
#     'path_GT': "/manual/",
#     'C1_name': 'Healthy',
#     'C2_name': 'Sclerotic'
# }

# RT_dict = {
#     'path_ds': original_dataset_path + list_of_dataset[2] + "/DATASET/",
#     'path_ks': original_dataset_path + list_of_dataset[2] + "/k-net+swin/TEST/",
#     'path_GT': "/manual/",
#     'C1_name': 'Healthy',
#     'C2_name': 'Atro'
# }

dataset_dict = {
    # 'Liver HE steatosis 2c': LS_2c_dict,
    'Liver HE steatosis 3c': LS_3c_dict
#     'Renal PAS glomeruli': RG_dict,
#     'Renal PAS tubuli': RT_dict
}


#%%

sha_ent = {
    'name': 'sha_ent_',
    'function': shannon_entropy_map
    }

bin_ent = {
    'name': 'bin_ent_',
    'function': binary_entropy_map
    }

std_sum = {
    'name': 'std_map_',
    'function': std_map
    }

uncertainty_metric = {
    'sha_ent' : sha_ent,
    'std_sum' : std_sum,
    'bin_ent' : bin_ent
    }

for p in uncertainty_metric:
    print(uncertainty_metric[p]['name'])

#%% NOME IMMAGINE, PATHS PER IMMAGINE
dataset = 'Liver HE steatosis 3c'
subset = 'val'
name = '1004546_22.png'
# name = '1004338_1.png'
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
    
original_image = np.array(Img.open(original_image_path))
GT_mask_orig = np.array(Img.open(GT_mask_path))/255
SI_mask_orig = np.array(Img.open(SI_mask_path))/255

GT_mask_C1, GT_mask_C2 = mask_splitter(GT_mask_orig)
SI_mask_C1, SI_mask_C2 = mask_splitter(SI_mask_orig)

DIM = np.shape(GT_mask_orig)
N = 20
c = 3

softmax_matrix_PERT = softmax_matrix_gen(
    PERT_path_20_softmax, DIM, c, N)

#%% Creazione Mask AVG e Mask Union
softmax_avg_PERT = np.mean(softmax_matrix_PERT, axis=-1)
mask_avg_PERT = np.argmax(softmax_avg_PERT,axis=-1)
mask_avg_C1_PERT, mask_avg_C2_PERT = mask_splitter(mask_avg_PERT/2)

mask_union_matrix = np.argmax(softmax_matrix_PERT, axis=2)
mask_union_PERT = np.sum(mask_union_matrix, axis=-1)
mask_union_PERT[mask_union_PERT>0] = 1

#%%
original_image_path = LS_2c_dict['path_ds'] + \
                subset+"/image/" + name
GT_mask_path = LS_2c_dict['path_ds'] + \
    subset+"/manual/" + name
SI_mask_path = LS_2c_dict['path_ks'] + \
    "RESULTS/"+subset+"/mask/"+name

PERT_path_20_seg_2c = LS_2c_dict['path_ks'] + \
    '/RESULTS_perturbation/'+subset+'/mask/' + name + "/"
PERT_path_20_softmax_2c = LS_2c_dict['path_ks'] + \
    '/RESULTS_perturbation/'+subset+'/softmax/' + name + "/"

MC_path_20_seg_2c = LS_2c_dict['path_ks'] + \
    '/RESULTS_MC/'+subset+'/mask/' + name + "/"
MC_path_20_softmax_2c = LS_2c_dict['path_ks'] + \
    '/RESULTS_MC/'+subset+'/softmax/' + name + "/"

save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V5/"+dataset+"/"+subset+"/"+name+"/"
if not os.path.isdir(save_path):
    os.makedirs(save_path)

original_image_2c = np.array(Img.open(original_image_path))
GT_mask_orig_2c = np.array(Img.open(GT_mask_path))/255
SI_mask_orig_2c = np.array(Img.open(SI_mask_path))/255

DIM = np.shape(GT_mask_orig_2c)
N = 20
c = 2

softmax_matrix_PERT_2c = softmax_matrix_gen(
    PERT_path_20_softmax, DIM, c, N)

softmax_matrix_PERT_2c[:,:,1,:][GT_mask_C1>0] = 0
bin_ent_map_PERT_2c = binary_entropy_map(softmax_matrix_PERT_2c[:,:,1,:])
BRUM = copy.deepcopy(bin_ent_map_PERT_2c)

#%% Regionprops FARE ANCHE SU SINGLE INFERENCE PER SFIZIO
# MASK_TO_USE_TO_REDUCE = SI_mask_C2
MASK_TO_USE_TO_REDUCE = mask_avg_C2_PERT
label_image_PERT = measure.label(MASK_TO_USE_TO_REDUCE)
n_objects_PERT = label_image_PERT.max()
masks_matrix_PERT = np.zeros([label_image_PERT.shape[0],label_image_PERT.shape[1],n_objects_PERT])
unc_map_matrix_PERT = np.copy(masks_matrix_PERT)
tau_array_PERT = []
for i in range(n_objects_PERT):
    if i == 0: continue
    current_mask = np.copy(label_image_PERT)
    current_mask[current_mask!=i] = 0
    max_mask = np.max(current_mask)
    if max_mask == 0: max_mask = 1
    current_mask = current_mask/max_mask
    masks_matrix_PERT[:,:,i] = current_mask
    unc_map_matrix_PERT = current_mask*BRUM
    tau_i = np.nanmean(unc_map_matrix_PERT[unc_map_matrix_PERT>0])
    tau_array_PERT.append(tau_i)
    del current_mask
    
#%% Loop con threshold
array_PERT = copy.deepcopy(tau_array_PERT)
array_PERT = np.array(array_PERT)
th_range = np.array(range(0,100,1))/100
# th_range = np.array(range(0,10,1))/10

# 3 strade sono:
# dice mask_th mask_avg 
# dice mask_th SI
# dice mask_th mask_unione

dice_array_PERT_mask_avg=[]
dice_array_PERT_SI=[]
dice_array_PERT_mask_unione=[]
counter = 0
masks_of_masks_PERT = np.zeros((DIM[0],DIM[1],len(th_range)))
SI_dice = dice(SI_mask_orig_2c,GT_mask_orig_2c)

for th in th_range:
    array_PERT_temp = copy.deepcopy(array_PERT)
    array_PERT_temp[array_PERT>th] = 0
    masks_matrix_temp = masks_matrix_PERT[:,:,np.where(array_PERT_temp>0)[0]]
    mask_th = np.sum(masks_matrix_temp, axis=-1)
    masks_of_masks_PERT[:,:,counter] = mask_th
    
    SI_reduced = np.zeros_like(SI_mask_C2)
    SI_reduced[mask_th>0] = SI_mask_C2[mask_th>0]
    
    mask_avg_reduced = np.zeros_like(mask_avg_C2_PERT)
    mask_avg_reduced[mask_th>0] = mask_avg_C2_PERT[mask_th>0]
    
    mask_unione_reduced = np.zeros_like(mask_union_PERT)
    mask_unione_reduced[mask_th>0] = mask_union_PERT[mask_th>0]
    
    dice_th_mask_avg = dice(mask_avg_reduced,GT_mask_orig_2c)
    dice_array_PERT_mask_avg.append(dice_th_mask_avg)
    
    dice_th_SI = dice(SI_reduced,GT_mask_orig_2c)
    dice_array_PERT_SI.append(dice_th_SI)
    
    dice_th_mask_unione = dice(mask_unione_reduced,GT_mask_orig_2c)
    dice_array_PERT_mask_unione.append(dice_th_mask_unione)
    
    if (dice_th_mask_unione>SI_dice or dice_th_mask_avg>SI_dice or dice_th_SI>SI_dice):
        plt.figure(figsize=(15,15))
        plt.suptitle("TH = " + str(th))
        plt.subplot(221)
        plt.title("Ground Truth")
        plt.imshow(GT_mask_orig_2c)
        plt.subplot(222)
        plt.title("MASK_AVG_REDUCED - dice " + str(dice_th_mask_avg))
        plt.imshow(mask_avg_reduced)
        plt.subplot(223)
        plt.title("mask_unione_reduced - dice " + str(dice_th_mask_unione))
        plt.imshow(mask_unione_reduced)
        plt.subplot(224)
        plt.title("SI_reduced - dice " + str(dice_th_SI))
        plt.imshow(SI_reduced)    
        plt.savefig(save_path + "TH=" + str(th) + ".png")
        plt.close()
        print("Immagine " + name + " ha cose belle")
    counter += 1 
    


dice_array_PERT_mask_avg = np.array(dice_array_PERT_mask_avg)
miglioramento_PERT_mask_avg = np.where(dice_array_PERT_mask_avg>SI_dice)[0]
# print(miglioramento_PERT_mask_avg)

dice_array_PERT_SI = np.array(dice_array_PERT_SI)
miglioramento_PERT_SI = np.where(dice_array_PERT_SI>SI_dice)[0]
# print(miglioramento_PERT_SI)

dice_array_PERT_mask_unione = np.array(dice_array_PERT_mask_unione)
miglioramento_PERT_mask_unione = np.where(dice_array_PERT_mask_unione>SI_dice)[0]
# print(miglioramento_PERT_mask_unione)

#%% PLOT
plt.figure(figsize=(10,10))
plt.plot(th_range,dice_array_PERT_mask_avg, label="Dice Score per Threshold (PERT) - Mask Avg")
plt.axhline(SI_dice, color='r', linestyle='--', label='Single Inference Dice')
plt.title("Dice x Threshold")
plt.xlabel('Threshold')
plt.ylabel('Dice Score')
plt.legend()

plt.figure(figsize=(10,10))
plt.plot(th_range,dice_array_PERT_SI, label="Dice Score per Threshold (PERT) - Single Inference")
plt.axhline(SI_dice, color='r', linestyle='--', label='Single Inference Dice')
plt.title("Dice x Threshold")
plt.xlabel('Threshold')
plt.ylabel('Dice Score')
plt.legend()

plt.figure(figsize=(10,10))
plt.plot(th_range,dice_array_PERT_mask_unione, label="Dice Score per Threshold (PERT) - Union Mask")
plt.axhline(SI_dice, color='r', linestyle='--', label='Single Inference Dice')
plt.title("Dice x Threshold")
plt.xlabel('Threshold')
plt.ylabel('Dice Score')
plt.legend()







