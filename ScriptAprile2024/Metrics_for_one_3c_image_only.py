# # -*- coding: utf-8 -*-
# """
# Created on Wed Apr 24 15:02:55 2024

# @author: willy
# """

#%% Import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Img
from tqdm import tqdm

#%% Primary functions
def dice(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()

    if TP==0 and FN==0 and FP==0:
        dice_ind = np.nan
    else:
        dice_ind = 2*TP/(2*TP+FP+FN)
    return dice_ind

#%%

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


#%% Map generator functions

# Binary Entropy Map
def binary_entropy_map(softmax_matrix):
    p_mean = np.mean(softmax_matrix, axis=2)
    p_mean[p_mean==0] = 1e-8 
    p_mean[p_mean==1] = 1-1e-8
    HB_pert = -(np.multiply(p_mean,np.log2(p_mean)) + np.multiply((1-p_mean),np.log2(1-p_mean)))
    return HB_pert
    

# Shannon Entropy Map
def shannon_entropy_map(softmax_matrix):
    shape = np.shape(softmax_matrix)
    shannon_entropy_map = np.zeros([shape[0],shape[1]]).astype(float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ij_vector = softmax_matrix[i,j,:]
            shannon_entropy_map[i,j] = -(np.multiply(ij_vector[ij_vector!=0], np.log2(ij_vector[ij_vector!=0]))).sum()
    return shannon_entropy_map

# Std Map
def std_map(softmax_matrix): return np.nanstd(softmax_matrix, axis=2)

# Dice Mat
def dice_mat_map(seg_matrix,N):
    dice_mat = -np.ones((N,N))
    for i in range(N):
        for j in range(i+1,N):
            dice_mat[i,j] = dice(seg_matrix[:,:,i],seg_matrix[:,:,j])
    dice_mat[dice_mat<0] = np.nan
    return dice_mat

#%% Metric calculator functions

# Binary Entropy Value given one map 2 classes
def bin_ent_sum_2c(bin_ent_map): return np.nansum(bin_ent_map)

# Binary Entropy Value given one map 3 classes
def bin_ent_sum_3c(bin_ent_map_c1_value,bin_ent_map_c2_value): return np.mean([bin_ent_map_c1_value,bin_ent_map_c2_value])

# Shannon Entropy Value given one map 2 classes
def sha_ent_sum_2c(sha_ent_map): return np.nansum(sha_ent_map)

# Shannon Entropy Value given one map 3 classes 
def sha_ent_sum_3c(sha_ent_map_c1_value,sha_ent_map_c2_value): return np.mean([sha_ent_map_c1_value,sha_ent_map_c2_value])

# Std map value given one map 2 classes
def std_sum_2c(std_map): return np.nansum(std_map)

# Std map value given one map 3 classes
def std_sum_3c(std_map_c1_value,std_map_c2_value): return np.mean([std_map_c1_value,std_map_c2_value])

# Dice mat value given one map 2 classes
def dice_mat_std_2c(dice_mat): return np.nanstd(dice_mat)
def dice_mat_mean_2c(dice_mat): return np.nanmean(dice_mat)

# Dice mat value given one map 3 classes
def dice_mat_std_3c(dice_mat_c1_value,dice_mat_c2_value): return np.mean([dice_mat_c1_value,dice_mat_c2_value])
def dice_mat_mean_3c(dice_mat_c1_value,dice_mat_c2_value): return np.mean([dice_mat_c1_value,dice_mat_c2_value])

# BC given one map 3 classes

# KL given one map 3 classes


#%% Paths
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

LS_2c_path_GT = "/manual/"
LS_3c_path_GT = "/manual/"

RG_3c_path_GT = "/manual/"
RG_3c_path_GT_1 = "/manual_healthy/"
RG_3c_path_GT_2 = "/manual_sclerotic/"

RT_3c_path_GT = "/manual/"
RT_3c_path_GT_1 = "/manual_healthy/"
RT_3c_path_GT_2 = "/manual_atro/"

dataset_for_loop = [LS_2c_path_ds,LS_3c_path_ds,RG_3c_path_ds,RT_3c_path_ds]

#%%
LS_3c_dict = {
    'path_ds': LS_3c_path_ds,
    'path_ks': LS_3c_path_ks,
    'path_GT': LS_3c_path_GT,
    'C1_name': 'Class 1',
    'C2_name': 'Class 2'
    }

RG_dict = {
    'path_ds': original_dataset_path + list_of_dataset[1] +"/DATASET/",
    'path_ks': original_dataset_path + list_of_dataset[1] +"/k-net+swin/TEST/",
    'path_GT': RG_3c_path_GT,
    'path_GT_1': RG_3c_path_GT_1,
    'path_GT_2': RG_3c_path_GT_2,
    'C1_name': 'Healthy',
    'C2_name': 'Sclerotic'
    }

RT_dict = {
    'path_ds': original_dataset_path + list_of_dataset[2] +"/DATASET/",
    'path_ks': original_dataset_path + list_of_dataset[2] +"/k-net+swin/TEST/",
    'path_GT': RT_3c_path_GT,
    'path_GT_1': RT_3c_path_GT_1,
    'path_GT_2': RT_3c_path_GT_2,
    'C1_name': 'Healthy',
    'C2_name': 'Atro'
    }

dataset_dict = {
    # 'Liver HE steatosis 2c': LS_2c_dict,
    'Liver HE steatosis 3c': LS_3c_dict,
    'Renal PAS glomeruli': RG_dict,
    'Renal PAS tubuli': RT_dict
    }

#%%
# for dataset in dataset_for_loop:
for dataset in dataset_dict:
    for subset in ['val','test']:
        image_list = os.listdir(dataset_dict[dataset]['path_ds']+subset+"/"+dataset_dict[dataset]['path_GT'])
        metrics_matrix = []
        image_counter = 0
        metrics_save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V2/"+dataset+"/"+subset+"/"
        for name in tqdm(image_list):
            original_image_path = dataset_dict[dataset]['path_ds']+subset+"/image/" + name
            GT_mask_path = dataset_dict[dataset]['path_ds']+subset+"/manual/" + name
            SI_path = dataset_dict[dataset]['path_ks']+"RESULTS/"+subset+"/mask/"+name
            PERT_path_20_seg = dataset_dict[dataset]['path_ks']+'/RESULTS_perturbation/'+subset+'/mask/' + name + "/"
            PERT_path_20_soft = dataset_dict[dataset]['path_ks']+'/RESULTS_perturbation/'+subset+'/softmax/' + name + "/"
            save_path = "D:/DATASET_Tesi_Marzo2024_RESULTS_V2/"+dataset+"/"+subset+"/"+name+"/"
            
            if not os.path.isdir(save_path): os.makedirs(save_path)

            original_image = Img.open(original_image_path)
            GT_mask = np.array(Img.open(GT_mask_path))/255
            SI_mask = np.array(Img.open(SI_path))/255
            
            GT_mask_values = np.unique(GT_mask)
            SI_mask_values = np.unique(SI_mask)
            standard_values = [0.0,0.5,1.0]
            
            classes_GT_mask = len(GT_mask_values)
            classes_SI_mask = len(SI_mask_values)
            
            GT_mask_standard = np.copy(GT_mask)
            for i in range(classes_GT_mask):
                GT_mask_standard[GT_mask_standard==GT_mask_values[i]] = standard_values[i]
            # if np.max(GT_mask_standard)==0.5: GT_mask_standard[GT_mask_standard==0.5] = 1
            
            SI_mask_standard = np.copy(SI_mask)
            for i in range(classes_SI_mask):
                SI_mask_standard[SI_mask_standard==SI_mask_values[i]] = standard_values[i]
            if np.max(SI_mask_standard)==0.5: SI_mask_standard[SI_mask_standard==0.5] = 1
            
            GT_mask_c1 = np.copy(GT_mask_standard)
            GT_mask_c1[GT_mask_c1==1] = 0
            GT_mask_c1[GT_mask_c1==0.5] = 1
            # GT_mask_c1_weight = np.nansum(GT_mask_c1)
            
            GT_mask_c2 = np.copy(GT_mask_standard)
            GT_mask_c2[GT_mask_c2==0.5] = 0
            # GT_mask_c2_weight = np.nansum(GT_mask_c2)
            
            SI_mask_c1 = np.copy(SI_mask_standard)
            SI_mask_c1[SI_mask_c1==1] = 0
            SI_mask_c1[SI_mask_c1==0.5] = 1
            # SI_mask_c1_weight = np.nansum(SI_mask_c1)
            
            SI_mask_c2 = np.copy(SI_mask_standard)
            SI_mask_c2[SI_mask_c2==0.5] = 0
            # SI_mask_c2_weight = np.nansum(SI_mask_c2)
            
            
            # Dice calculation
            dice_real_value_c1 = dice(SI_mask_c1,GT_mask_c1)
            dice_real_value_c2 = dice(SI_mask_c2,GT_mask_c2)
            # dice_real_value_mean = (dice_real_value_c1*SI_mask_c1_weight+dice_real_value_c2*SI_mask_c2_weight)/(SI_mask_c1_weight+SI_mask_c2_weight)
            
            DIM = np.shape(GT_mask)
            N = 20
            c = 3 
            
            # softmax e seg matrix generation
            softmax_matrix_PERT = softmax_matrix_gen(PERT_path_20_soft, DIM, c, N)
            seg_matrix_PERT = seg_matrix_gen(PERT_path_20_seg, DIM, N)
            temp_c1 = np.copy(seg_matrix_PERT)
            temp_c1[temp_c1==1] = 0
            temp_c1[temp_c1==0.5] = 1
            temp_c2 = np.copy(seg_matrix_PERT)
            temp_c2[temp_c2<1] = 0
            seg_matrix_PERT_c1 = np.copy(temp_c1)
            seg_matrix_PERT_c2 = np.copy(temp_c2)
            
            # uncertainty maps generation
            bin_ent_map_c1 = binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
            sha_ent_map_c1 = shannon_entropy_map(softmax_matrix_PERT[:,:,1,:])
            std_map_c1 = std_map(softmax_matrix_PERT[:,:,1,:])
            dice_mat_map_c1 = dice_mat_map(seg_matrix_PERT_c1, N)
        
            bin_ent_map_c2 = binary_entropy_map(softmax_matrix_PERT[:,:,2,:])
            sha_ent_map_c2 = shannon_entropy_map(softmax_matrix_PERT[:,:,2,:])
            std_map_c2 = std_map(softmax_matrix_PERT[:,:,2,:])
            dice_mat_map_c2 = dice_mat_map(seg_matrix_PERT_c2, N)
            
            
            plt.figure(figsize=(20,10))
        
            plt.subplot(251)
            plt.imshow(original_image)
            plt.title('Original Image')
        
            plt.subplot(252)
            plt.imshow(bin_ent_map_c1)
            plt.colorbar()
            plt.title('Binary Entropy ' + dataset_dict[dataset]['C1_name'])
        
            plt.subplot(253)
            plt.imshow(sha_ent_map_c1)
            plt.colorbar()
            plt.title('Shannon Entropy '+ dataset_dict[dataset]['C1_name'])
        
            plt.subplot(254)
            plt.imshow(std_map_c1)
            plt.colorbar()
            plt.title('Std Map ' + dataset_dict[dataset]['C1_name'])
        
            plt.subplot(255)
            plt.imshow(dice_mat_map_c1)
            plt.colorbar()
            plt.title('Dice Mat ' + dataset_dict[dataset]['C1_name'])
        
            plt.subplot(256)
            plt.imshow(GT_mask)
            plt.title('Ground Truth')
        
            plt.subplot(257)
            plt.imshow(bin_ent_map_c2)
            plt.colorbar()
            plt.title('Binary Entropy ' + dataset_dict[dataset]['C2_name'])
        
            plt.subplot(258)
            plt.imshow(sha_ent_map_c2)
            plt.colorbar()
            plt.title('Shannon Entropy ' + dataset_dict[dataset]['C2_name'])
        
            plt.subplot(259)
            plt.imshow(std_map_c2)
            plt.colorbar()
            plt.title('Std Map ' + dataset_dict[dataset]['C2_name'])
        
            plt.subplot(2,5,10)
            plt.imshow(dice_mat_map_c2)
            plt.colorbar()
            plt.title('Dice Mat ' + dataset_dict[dataset]['C2_name'])
            
            plt.savefig(save_path)
            plt.close()
            
            # uncertainty metrics calculation
            bin_ent_map_c1_value = bin_ent_sum_2c(bin_ent_map_c1)
            bin_ent_map_c2_value = bin_ent_sum_2c(bin_ent_map_c2)
            bin_ent_map_mean_value = bin_ent_sum_3c(bin_ent_map_c1_value, bin_ent_map_c2_value)
        
            sha_ent_map_c1_value = sha_ent_sum_2c(sha_ent_map_c1)
            sha_ent_map_c2_value = sha_ent_sum_2c(sha_ent_map_c2)
            sha_ent_map_mean_value = sha_ent_sum_3c(sha_ent_map_c1_value, sha_ent_map_c2_value)
        
            std_map_c1_value = std_sum_2c(std_map_c1)
            std_map_c2_value = std_sum_2c(std_map_c2)
            std_map_mean_value = std_sum_3c(std_map_c1_value, std_map_c2_value)
        
            dice_map_c1_std = dice_mat_std_2c(dice_mat_map_c1)
            dice_map_c2_std = dice_mat_std_2c(dice_mat_map_c2)
            dice_map_mean_std = dice_mat_std_3c(dice_map_c1_std, dice_map_c2_std)
        
            dice_map_c1_mean = dice_mat_mean_2c(dice_mat_map_c1)
            dice_map_c2_mean = dice_mat_mean_2c(dice_mat_map_c2)
            dice_map_mean_mean = dice_mat_mean_3c(dice_map_c1_mean, dice_map_c2_mean)
            
            metrics_row = [name,
                           bin_ent_map_c1_value,
                           bin_ent_map_c2_value,
                           bin_ent_map_mean_value,
                           sha_ent_map_c1_value,
                           sha_ent_map_c2_value,
                           sha_ent_map_mean_value,
                           std_map_c1_value,
                           std_map_c2_value,
                           std_map_mean_value,
                           dice_map_c1_std,
                           dice_map_c2_std,
                           dice_map_mean_std,
                           dice_map_c1_mean,
                           dice_map_c2_mean,
                           dice_map_mean_mean,
                           dice_real_value_c1, #16
                           dice_real_value_c2, #17
                           # dice_real_value_mean
                           ]
            
            metrics_matrix.append(metrics_row)
            image_counter += 1
            # if image_counter==3: break
            
        metrics_df = pd.DataFrame(metrics_matrix, columns=['name',
                       'bin_ent_map_c1_value',
                       'bin_ent_map_c2_value',
                       'bin_ent_map_mean_value',
                       'sha_ent_map_c1_value',
                       'sha_ent_map_c2_value',
                       'sha_ent_map_mean_value',
                       'std_map_c1_value',
                       'std_map_c2_value',
                       'std_map_mean_value',
                       'dice_map_c1_std',
                       'dice_map_c2_std',
                       'dice_map_mean_std',
                       'dice_map_c1_mean',
                       'dice_map_c2_mean',
                       'dice_map_mean_mean',
                       'dice_real_value_c1',
                       'dice_real_value_c2',
                       # 'dice_real_value_mean'
                       ])
        
        metrics_df.to_csv(metrics_save_path+"metrics.csv",index=False)


















# #%%
# metrics_temp = np.copy(metrics_matrix)
# metrics_temp_1 = metrics_temp[metrics_temp[:,16]!='0.0']
# metrics_temp_2 = metrics_temp[metrics_temp[:,17]!='0.0']

# metrics_df_clean_1 = pd.DataFrame(metrics_temp_1, columns=['name',
#                'bin_ent_map_c1_value',
#                'bin_ent_map_c2_value',
#                'bin_ent_map_mean_value',
#                'sha_ent_map_c1_value',
#                'sha_ent_map_c2_value',
#                'sha_ent_map_mean_value',
#                'std_map_c1_value',
#                'std_map_c2_value',
#                'std_map_mean_value',
#                'dice_map_c1_std',
#                'dice_map_c2_std',
#                'dice_map_mean_std',
#                'dice_map_c1_mean',
#                'dice_map_c2_mean',
#                'dice_map_mean_mean',
#                'dice_real_value_c1',
#                'dice_real_value_c2',
#                'dice_real_value_mean'
#                ])

# metrics_df_clean_2 = pd.DataFrame(metrics_temp_2, columns=['name',
#                'bin_ent_map_c1_value',
#                'bin_ent_map_c2_value',
#                'bin_ent_map_mean_value',
#                'sha_ent_map_c1_value',
#                'sha_ent_map_c2_value',
#                'sha_ent_map_mean_value',
#                'std_map_c1_value',
#                'std_map_c2_value',
#                'std_map_mean_value',
#                'dice_map_c1_std',
#                'dice_map_c2_std',
#                'dice_map_mean_std',
#                'dice_map_c1_mean',
#                'dice_map_c2_mean',
#                'dice_map_mean_mean',
#                'dice_real_value_c1',
#                'dice_real_value_c2',
#                'dice_real_value_mean'
#                ])

# #%%
# plt.scatter(np.array(metrics_df_clean_1['bin_ent_map_c1_value']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))
# r_b1 = np.corrcoef(np.array(metrics_df_clean_1['bin_ent_map_c1_value']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df_clean_2['bin_ent_map_c2_value']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))
# r_b2 = np.corrcoef(np.array(metrics_df_clean_2['bin_ent_map_c2_value']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))


# plt.scatter(np.array(metrics_df_clean_1['sha_ent_map_c1_value']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))
# r_sh1 = np.corrcoef(np.array(metrics_df_clean_1['sha_ent_map_c1_value']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df_clean_2['sha_ent_map_c2_value']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))
# r_sh2 = np.corrcoef(np.array(metrics_df_clean_2['sha_ent_map_c2_value']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))


# plt.scatter(np.array(metrics_df_clean_1['std_map_c1_value']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))
# r_std1 = np.corrcoef(np.array(metrics_df_clean_1['std_map_c1_value']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df_clean_2['std_map_c2_value']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))
# r_std2 = np.corrcoef(np.array(metrics_df_clean_2['std_map_c2_value']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))


# plt.scatter(np.array(metrics_df_clean_1['dice_map_c1_std']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))
# r_std_dice1 = np.corrcoef(np.array(metrics_df_clean_1['dice_map_c1_std']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df_clean_2['dice_map_c2_std']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))
# r_std_dice2 = np.corrcoef(np.array(metrics_df_clean_2['dice_map_c2_std']).astype(float),np.array(metrics_df_clean_2['dice_real_value_c2']).astype(float))

# #%%
# plt.scatter(np.array(metrics_df['bin_ent_map_c1_value']).astype(float),np.array(metrics_df['dice_real_value_c1']).astype(float))
# r_b1 = np.corrcoef(np.array(metrics_df['bin_ent_map_c1_value']).astype(float),np.array(metrics_df['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df['bin_ent_map_c2_value']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))
# r_b2 = np.corrcoef(np.array(metrics_df['bin_ent_map_c2_value']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))


# plt.scatter(np.array(metrics_df['sha_ent_map_c1_value']).astype(float),np.array(metrics_df['dice_real_value_c1']).astype(float))
# r_sh1 = np.corrcoef(np.array(metrics_df['sha_ent_map_c1_value']).astype(float),np.array(metrics_df['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df['sha_ent_map_c2_value']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))
# r_sh2 = np.corrcoef(np.array(metrics_df['sha_ent_map_c2_value']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))


# plt.scatter(np.array(metrics_df['std_map_c1_value']).astype(float),np.array(metrics_df['dice_real_value_c1']).astype(float))
# r_std1 = np.corrcoef(np.array(metrics_df['std_map_c1_value']).astype(float),np.array(metrics_df['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df['std_map_c2_value']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))
# r_std2 = np.corrcoef(np.array(metrics_df['std_map_c2_value']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))


# plt.scatter(np.array(metrics_df_clean_1['dice_map_c1_std']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))
# r_std_dice1 = np.corrcoef(np.array(metrics_df_clean_1['dice_map_c1_std']).astype(float),np.array(metrics_df_clean_1['dice_real_value_c1']).astype(float))

# plt.scatter(np.array(metrics_df['dice_map_c2_std']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))
# r_std_dice2 = np.corrcoef(np.array(metrics_df['dice_map_c2_std']).astype(float),np.array(metrics_df['dice_real_value_c2']).astype(float))


















#%%
# softmax_matrix_PERT = softmax_matrix_gen(PERT_path_20_soft, DIM, c, N)
# seg_matrix_PERT = seg_matrix_gen(PERT_path_20_seg, DIM, N)
# temp_c1 = np.copy(seg_matrix_PERT)
# temp_c1[temp_c1==1] = 0
# temp_c1[temp_c1==0.5] = 1
# temp_c2 = np.copy(seg_matrix_PERT)
# temp_c2[temp_c2<1] = 0
# seg_matrix_PERT_c1 = np.copy(temp_c1)
# seg_matrix_PERT_c2 = np.copy(temp_c2)

# bin_ent_map_c1 = binary_entropy_map(softmax_matrix_PERT[:,:,1,:])
# sha_ent_map_c1 = shannon_entropy_map(softmax_matrix_PERT[:,:,1,:])
# std_map_c1 = std_map(softmax_matrix_PERT[:,:,1,:])
# dice_mat_map_c1 = dice_mat_map(seg_matrix_PERT_c1, N)

# bin_ent_map_c2 = binary_entropy_map(softmax_matrix_PERT[:,:,2,:])
# sha_ent_map_c2 = shannon_entropy_map(softmax_matrix_PERT[:,:,2,:])
# std_map_c2 = std_map(softmax_matrix_PERT[:,:,2,:])
# dice_mat_map_c2 = dice_mat_map(seg_matrix_PERT_c2, N)

# #%%
# plt.figure(figsize=(20,10))

# plt.subplot(251)
# plt.imshow(original_image)
# plt.title('Original Image')

# plt.subplot(252)
# plt.imshow(bin_ent_map_c1)
# plt.colorbar()
# plt.title('Binary Entropy C1')

# plt.subplot(253)
# plt.imshow(sha_ent_map_c1)
# plt.colorbar()
# plt.title('Shannon Entropy C1')

# plt.subplot(254)
# plt.imshow(std_map_c1)
# plt.colorbar()
# plt.title('Std Map C1')

# plt.subplot(255)
# plt.imshow(dice_mat_map_c1)
# plt.colorbar()
# plt.title('Dice Mat C1')

# plt.subplot(256)
# plt.imshow(GT_mask)
# plt.title('Ground Truth')

# plt.subplot(257)
# plt.imshow(bin_ent_map_c2)
# plt.colorbar()
# plt.title('Binary Entropy C2')

# plt.subplot(258)
# plt.imshow(sha_ent_map_c2)
# plt.colorbar()
# plt.title('Shannon Entropy C2')

# plt.subplot(259)
# plt.imshow(std_map_c2)
# plt.colorbar()
# plt.title('Std Map C2')

# plt.subplot(2,5,10)
# plt.imshow(dice_mat_map_c2)
# plt.colorbar()
# plt.title('Dice Mat C2')

# plt.show()
# #%%

# bin_ent_map_c1_value = bin_ent_sum_2c(bin_ent_map_c1)
# bin_ent_map_c2_value = bin_ent_sum_2c(bin_ent_map_c2)
# bin_ent_map_mean_value = bin_ent_sum_3c(bin_ent_map_c1_value, bin_ent_map_c2_value)

# sha_ent_map_c1_value = sha_ent_sum_2c(sha_ent_map_c1)
# sha_ent_map_c2_value = sha_ent_sum_2c(sha_ent_map_c2)
# sha_ent_map_mean_value = sha_ent_sum_3c(sha_ent_map_c1_value, sha_ent_map_c2_value)

# std_map_c1_value = std_sum_2c(std_map_c1)
# std_map_c2_value = std_sum_2c(std_map_c2)
# std_map_mean_value = std_sum_3c(std_map_c1_value, std_map_c2_value)

# dice_map_c1_std = dice_mat_std_2c(dice_mat_map_c1)
# dice_map_c2_std = dice_mat_std_2c(dice_mat_map_c2)
# dice_map_mean_std = dice_mat_std_3c(dice_map_c1_std, dice_map_c2_std)

# dice_map_c1_mean = dice_mat_mean_2c(dice_mat_map_c1)
# dice_map_c2_mean = dice_mat_mean_2c(dice_mat_map_c2)
# dice_map_mean_mean = dice_mat_mean_3c(dice_map_c1_mean, dice_map_c2_mean)




