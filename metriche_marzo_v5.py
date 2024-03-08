#%% import
import os
import numpy as np
import pandas as pd
import metrics_v4 as m
import matplotlib.pyplot as plt
# import cv2
import PIL.Image as Img
from tqdm import tqdm
from scipy import stats

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error 

#%%
def cv_map(softmax_matrix):
    shape = np.shape(softmax_matrix)
    cv_map = np.zeros([shape[0],shape[1]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            ij_vector = softmax_matrix[i,j,1,:]
            sum_ij_vector = np.nansum(ij_vector)
            mu = sum_ij_vector/len(ij_vector)
            std = np.nanstd(ij_vector)
            cv_map[i,j] = std/mu
    return cv_map

#%%
DIM = [416,416]
GT_seg_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/DATASET/test/manual/"
softmax_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/softmax/"
seg_single_inference_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS/test/mask/"
seg_MC_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/mask/"

#%%
i = 0
x_metrica_cv_map = []
x_metrica_dice_multi = []
for GT_seg_name in tqdm(os.listdir(GT_seg_dir)):
    directory_seg_MC = seg_MC_dir+GT_seg_name +"/"
    directory_softmax_MC = softmax_dir+GT_seg_name+"/"
    softmax_matrix = np.zeros((DIM[0],DIM[1],2,20),dtype=np.float32)
    seg_matrix = np.zeros((DIM[0],DIM[1],20))
    MC_softmax_list = os.listdir(directory_softmax_MC)
    MC_seg_list = os.listdir(directory_seg_MC)
    
    GT_mask_path = GT_seg_dir + GT_seg_name
    GT_mask_temp = Img.open(GT_mask_path)
    GT_mask = np.array(GT_mask_temp)/255
    
    # Creo softmax_matrix
    counter = 0
    for name in MC_softmax_list:
        st1 = np.float32((np.load(directory_softmax_MC + name)['softmax'])[:,:,1])
        st0 = np.float32((np.load(directory_softmax_MC + name)['softmax'])[:,:,0])
        softmax_matrix[:,:,0,counter] = np.copy(st0)
        softmax_matrix[:,:,1,counter] = np.copy(st1)
        counter += 1
    
    # Creo seg_matrix
    counter = 0
    for name in MC_seg_list:
        temp = Img.open(directory_seg_MC + name)
        seg_matrix[:,:,counter] = np.array(temp)/255
        counter += 1
        
    # Creo cv_map
    cv_map_temp = cv_map(softmax_matrix)
    GT_cv = cv_map_temp * GT_mask
    x_metrica_cv_map.append(100*(GT_cv>0).sum()/(416*416))
    
    dice_mat_temp = -np.ones((20,20))
    for i in range(20):
        for j in range(i+1,20):
            dice_mat_temp[i,j] = m.dice(seg_matrix[:,:,i],seg_matrix[:,:,j])
    dice_mat_temp[dice_mat_temp<0] = np.nan
    
    x_metrica_dice_multi.append(np.nanstd(dice_mat_temp))
    # break

 #%%
# GT_mask_path = GT_seg_dir + GT_seg_name
# GT_mask_temp = Img.open(GT_mask_path)
# GT_mask = np.array(GT_mask_temp)/255
# dice_array = []
# for i in range(20):
#     actual_seg_MC = seg_matrix[:,:,i]
#     temp_dice = m.dice(actual_seg_MC,GT_mask)
#     dice_array.append(temp_dice)
    
# #%%
# shape = np.shape(softmax_matrix)
# cv_map = np.zeros([shape[0],shape[1]])
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         ij_vector = softmax_matrix[i,j,1,:]
#         sum_ij_vector = np.nansum(ij_vector)
#         mu = sum_ij_vector/len(ij_vector)
#         std = np.nanstd(ij_vector)
#         cv_map[i,j] = std/mu
        
# #%%
# plt.figure(figsize=(18,6))
# plt.subplot(131)
# plt.imshow(cv_map)
# plt.title("CV Map")
# plt.subplot(132)
# plt.imshow(GT_mask)
# plt.title("GT Mask")
# plt.subplot(133)
# plt.imshow(GT_mask*cv_map)
# plt.title("CV Map x GT Mask")
# plt.show()

# #%%
# seg_inters = np.copy(seg_matrix[:,:,0])
# for i in range(20):
#     seg_inters = seg_matrix[:,:,i]*seg_inters
    
# #%%
# plt.imshow(seg_inters)

# #%%
# dice_mat = np.zeros((20,20))
# for i in range(20):
#     for j in range(i+1,20):
#         dice_mat[i,j] = m.dice(seg_matrix[:,:,i],seg_matrix[:,:,j])
        
# #%%

#%% PLOT: x_metriche vs numero di immagini
