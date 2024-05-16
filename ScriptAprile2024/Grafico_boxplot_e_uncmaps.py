# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:50:34 2024

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

# def unc_maps_plots(
#         dataset,
#         subset,
#         name,
#         original_image,
#         GT_mask,
#         bin_ent_map1,
#         sha_ent_map1,
#         std_map1,
#         dice_mat1,
#         bin_ent_map2,
#         sha_ent_map2,
#         std_map2,
#         dice_mat2,
#         save_path
#         ):

#     plt.figure(figsize=(25,60))
#     plt.suptitle('IMAGE: '+name[:-4]+", DATASET: "+dataset+", SUBSET: "+subset)

#     plt.subplot(251)
#     plt.title("Original Image")
#     plt.imshow(original_image)

#     plt.subplot(252)
#     plt.title("Binary Entropy Map")
#     plt.imshow(bin_ent_map1)
#     plt.colorbar()

#     plt.subplot(253)
#     plt.title("Shannon Entropy Map")
#     plt.imshow(sha_ent_map1)
#     plt.colorbar()

#     plt.subplot(254)
#     plt.title("Std Map")
#     plt.imshow(std_map1)
#     plt.colorbar()

#     plt.subplot(255)
#     plt.title("Inter-dice Matrix")
#     plt.imshow(dice_mat1)
#     plt.colorbar()

#     plt.subplot(256)
#     plt.title("Ground Truth Mask")
#     plt.imshow(GT_mask)

#     plt.subplot(257)
#     plt.title("Binary Entropy Map")
#     plt.imshow(bin_ent_map2)
#     plt.colorbar()

#     plt.subplot(258)
#     plt.title("Shannon Entropy Map")
#     plt.imshow(sha_ent_map2)
#     plt.colorbar()

#     plt.subplot(259)
#     plt.title("Std Map")
#     plt.imshow(std_map2)
#     plt.colorbar()

#     plt.subplot(2,5,10)
#     plt.title("Inter-dice Matrix")
#     plt.imshow(dice_mat2)
#     plt.colorbar()

#     plt.savefig(save_path)
#     plt.close()


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
        save_path
):

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
        image_list = os.listdir(
            dataset_dict[dataset]['path_ds']+subset+"/"+dataset_dict[dataset]['path_GT'])
        subset_save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V2/"+dataset+"/"+subset+"/"

        mean_MC_dice_C1 = []
        mean_PERT_dice_C1 = []
        mean_MC_dice_C2 = []
        mean_PERT_dice_C2 = []
        SI_dice_array_C1 = []
        SI_dice_array_C2 = []

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

            save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V2/"+dataset+"/"+subset+"/"+name+"/"
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

            # %% BOXPLOT
            fig_save_path = save_path + "plots/"
            if not os.path.isdir(fig_save_path):
                os.makedirs(fig_save_path)

            dice_GTSI_C1 = dice(GT_mask_C1, SI_mask_C1)
            SI_dice_array_C1.append(dice_GTSI_C1)
            dice_MC_C1 = []
            dice_PERT_C1 = []
            TOT_dice_C1 = []
            seg_matrix_PERT_c1 = np.zeros((DIM[0], DIM[1], N))
            seg_matrix_MC_c1 = np.zeros((DIM[0], DIM[1], N))

            dice_GTSI_C2 = dice(GT_mask_C2, SI_mask_C2)
            SI_dice_array_C2.append(dice_GTSI_C2)
            dice_MC_C2 = []
            dice_PERT_C2 = []
            TOT_dice_C2 = []
            seg_matrix_PERT_c2 = np.zeros((DIM[0], DIM[1], N))
            seg_matrix_MC_c2 = np.zeros((DIM[0], DIM[1], N))

            seg_count = 0
            for n_sample in os.listdir(MC_path_20_seg):
                actual_seg_MC = np.array(Img.open(MC_path_20_seg+n_sample))/255
                seg_MC_C1, seg_MC_C2 = mask_splitter(actual_seg_MC)
                actual_seg_PERT = np.array(
                    Img.open(PERT_path_20_seg+n_sample))/255
                seg_PERT_C1, seg_PERT_C2 = mask_splitter(actual_seg_PERT)

                seg_matrix_PERT_c1[:, :, seg_count] = seg_PERT_C1
                seg_matrix_PERT_c2[:, :, seg_count] = seg_PERT_C2
                seg_matrix_MC_c1[:, :, seg_count] = seg_MC_C1
                seg_matrix_MC_c2[:, :, seg_count] = seg_MC_C2

                dice_MC1 = dice(GT_mask_C1, seg_MC_C1)
                dice_PERT1 = dice(GT_mask_C1, seg_PERT_C1)

                dice_MC_C1.append(dice_MC1)
                dice_PERT_C1.append(dice_PERT1)
                TOT_dice_C1.append(dice_MC1)
                TOT_dice_C1.append(dice_PERT1)

                dice_MC2 = dice(GT_mask_C2, seg_MC_C2)
                dice_PERT2 = dice(GT_mask_C2, seg_PERT_C2)

                dice_MC_C2.append(dice_MC2)
                dice_PERT_C2.append(dice_PERT2)
                TOT_dice_C2.append(dice_MC2)
                TOT_dice_C2.append(dice_PERT2)

                seg_count += 1

            dice_MC_C1 = np.array(dice_MC_C1)
            dice_PERT_C1 = np.array(dice_PERT_C1)
            TOT_dice_C1 = np.array(TOT_dice_C1)

            dice_MC_C2 = np.array(dice_MC_C2)
            dice_PERT_C2 = np.array(dice_PERT_C2)
            TOT_dice_C2 = np.array(TOT_dice_C2)

            boxplot_generator(dice_GTSI_C1, dice_MC_C1, dice_PERT_C1, TOT_dice_C1,
                              fig_save_path+"boxplot_C1.png", name, dataset, subset, 1)
            boxplot_generator(dice_GTSI_C2, dice_MC_C2, dice_PERT_C2, TOT_dice_C2,
                              fig_save_path+"boxplot_C2.png", name, dataset, subset, 2)
            # uncertainty maps generation
            bin_ent_map_c1_PERT = binary_entropy_map(
                softmax_matrix_PERT[:, :, 1, :])
            sha_ent_map_c1_PERT = shannon_entropy_map(
                softmax_matrix_PERT[:, :, 1, :])
            std_map_c1_PERT = std_map(softmax_matrix_PERT[:, :, 1, :])
            dice_mat_map_c1_PERT = dice_mat_map(seg_matrix_PERT_c1, N)

            bin_ent_map_c1_MC = binary_entropy_map(
                softmax_matrix_MC[:, :, 1, :])
            sha_ent_map_c1_MC = shannon_entropy_map(
                softmax_matrix_MC[:, :, 1, :])
            std_map_c1_MC = std_map(softmax_matrix_MC[:, :, 1, :])
            dice_mat_map_c1_MC = dice_mat_map(seg_matrix_MC_c1, N)

            bin_ent_map_c2_PERT = binary_entropy_map(
                softmax_matrix_PERT[:, :, 2, :])
            sha_ent_map_c2_PERT = shannon_entropy_map(
                softmax_matrix_PERT[:, :, 2, :])
            std_map_c2_PERT = std_map(softmax_matrix_PERT[:, :, 2, :])
            dice_mat_map_c2_PERT = dice_mat_map(seg_matrix_PERT_c2, N)

            bin_ent_map_c2_MC = binary_entropy_map(
                softmax_matrix_MC[:, :, 2, :])
            sha_ent_map_c2_MC = shannon_entropy_map(
                softmax_matrix_MC[:, :, 2, :])
            std_map_c2_MC = std_map(softmax_matrix_MC[:, :, 2, :])
            dice_mat_map_c2_MC = dice_mat_map(seg_matrix_MC_c2, N)
# %%
            unc_maps_plots(dataset, subset, name, original_image, GT_mask_orig, SI_mask_orig, bin_ent_map_c1_MC, sha_ent_map_c1_MC, std_map_c1_MC,
                           dice_mat_map_c1_MC, bin_ent_map_c2_MC, sha_ent_map_c2_MC, std_map_c2_MC, dice_mat_map_c2_MC, fig_save_path+"unc_maps_plots_MC.png")
            unc_maps_plots(dataset, subset, name, original_image, GT_mask_orig, SI_mask_orig, bin_ent_map_c1_PERT, sha_ent_map_c1_PERT, std_map_c1_PERT,
                           dice_mat_map_c1_PERT, bin_ent_map_c2_PERT, sha_ent_map_c2_PERT, std_map_c2_PERT, dice_mat_map_c2_PERT, fig_save_path+"unc_maps_plots_PERT.png")

            dice_MC_C1[np.isnan(dice_MC_C1)] = 0
            dice_MC_C2[np.isnan(dice_MC_C1)] = 0
            dice_PERT_C1[np.isnan(dice_MC_C1)] = 0
            dice_PERT_C2[np.isnan(dice_MC_C1)] = 0

            mean_MC_dice_C1.append(np.nanmean(dice_MC_C1))
            mean_MC_dice_C2.append(np.nanmean(dice_MC_C2))
            mean_PERT_dice_C1.append(np.nanmean(dice_PERT_C1))
            mean_PERT_dice_C2.append(np.nanmean(dice_PERT_C2))
            count_image += 1

        plt.close()
        plt.plot(np.arange(len(image_list)), mean_MC_dice_C1)
        plt.plot(np.arange(len(image_list)), mean_PERT_dice_C1)
        plt.plot(np.arange(len(image_list)),
                 SI_dice_array_C1, linewidth=1, color='r')
        plt.title("Mean Dice per dataset: " + dataset +
                  ", subset: " + subset + ", class 1")
        plt.legend(labels=['Monte Carlo', 'Perturbations',
                   'Dice value Single Inference'])
        plt.xlabel('Images')
        plt.savefig(subset_save_path+"Mean_Dice_C1")
        plt.close()

        plt.plot(np.arange(len(image_list)), mean_MC_dice_C2)
        plt.plot(np.arange(len(image_list)), mean_PERT_dice_C2)
        plt.plot(np.arange(len(image_list)),
                 SI_dice_array_C1, linewidth=1, color='r')
        plt.title("Mean Dice per dataset: " + dataset +
                  ", subset: " + subset + ", class 2")
        plt.legend(labels=['Monte Carlo', 'Perturbations',
                   'Dice value Single Inference'])
        plt.xlabel('Images')
        plt.savefig(subset_save_path+"Mean_Dice_C2")
        plt.close()
