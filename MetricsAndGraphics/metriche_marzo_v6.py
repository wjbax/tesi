# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:58:19 2024
.

METRICS MARCH 2024 V6

.
@author: willy
"""

#%% import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Img
from tqdm import tqdm

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

# solo uno strato della softmax può essere inserito in questa funzione
def entropy_map(softmax_matrix):
    shape = np.shape(softmax_matrix)
    entropy_map = np.zeros([shape[0],shape[1]])
    temp_softmax_matrix = softmax_matrix[:,:,1,:]
    for i in range(shape[0]):
        for j in range(shape[1]):
            ij_vector = temp_softmax_matrix[i,j,:]
            sum_ij_vector = sum(ij_vector)
            ent = np.zeros(len(ij_vector))
            for k in range(len(ij_vector)):
                pt_k = (temp_softmax_matrix[i,j,k])/sum_ij_vector
                if pt_k == 0:
                    ent[k] = np.nan
                else:
                    log_k = np.log2(pt_k)
                    ent[k] = pt_k*log_k
            entropy_map[i,j] = -np.nansum(ent)
    return entropy_map

def maxminscal(array):
    MASSIMO = np.nanmax(array)
    minimo = np.nanmin(array)
    return (array-minimo)/(MASSIMO-minimo)

# def dice(seg1,seg2):
#     Z = seg1+seg2
#     if np.sum(Z) == 0: return 0
#     halfnum = np.count_nonzero(seg1*seg2>0) 
#     denom = np.sum(Z)
#     dice_value = 2*halfnum/denom
#     return dice_value

def dice(mask_automatic,mask_manual):
    TP_mask = np.multiply(mask_automatic,mask_manual); TP = TP_mask.sum()
    FP_mask = np.subtract(mask_automatic.astype(int),TP_mask.astype(int)).astype(bool); FP = FP_mask.sum()
    FN_mask = np.subtract(mask_manual.astype(int),TP_mask.astype(int)).astype(bool); FN = FN_mask.sum()
    # TN_mask = np.multiply(~mask_automatic,~mask_manual); TN = TN_mask.sum()
    
    if TP==0 and FN==0 and FP==0:
        # jaccard_ind = np.nan
        dice_ind = np.nan
    else:
        # jaccard_ind = TP/(TP+FP+FN)
        dice_ind = 2*TP/(2*TP+FP+FN)
    return dice_ind

#%%
DIM = [416,416]
GT_seg_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/DATASET/test/manual/"
softmax_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/softmax/"
seg_single_inference_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS/test/mask/"
seg_MC_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/mask/"
original_image_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/DATASET/test/image/"
#%%
i = 0
x_metrica_cv_map = []
x_metrica_ent_map = []
x_metrica_dice_multi = []
mean_dice_for_image = []
image_tracker = []
x_metrica_cv_map_sum = []
x_metrica_ent_map_sum = []

dice_mat_tracker = -np.ones((20,20,50))
softmax_matrix_tracker = np.zeros((416,416,2,20,50))
seg_matrix_tracker = np.zeros((416,416,20,50))
counter_tracker = 0

for GT_seg_name in tqdm(os.listdir(GT_seg_dir)):
    original_imagepng = Img.open(original_image_dir+GT_seg_name)
    original_image = np.array(original_imagepng)/255
    image_tracker.append(GT_seg_name)
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
    x_metrica_cv_map_sum.append(100*(GT_cv).sum()/(416*416))
    
    # Creo ent_map
    ent_map_temp = entropy_map(softmax_matrix)
    GT_ent = ent_map_temp * GT_mask
    x_metrica_ent_map.append(100*(GT_ent>0).sum()/(416*416))
    x_metrica_ent_map_sum.append(100*(GT_ent).sum()/(416*416))
    
    # Creo dice_mat_temp
    dice_mat_temp = -np.ones((20,20))
    dice_array_GT = []
    for i in range(20):
        for j in range(i+1,20):
            dice_mat_temp[i,j] = dice(seg_matrix[:,:,i],seg_matrix[:,:,j])
        dice_array_GT.append(dice(seg_matrix[:,:,i],GT_mask))
    dice_mat_temp[dice_mat_temp<0] = np.nan
    
    x_metrica_dice_multi.append(np.nanstd(dice_mat_temp))
    mean_dice_for_image.append(np.mean(dice_array_GT))
    
    dice_mat_tracker[:,:,counter_tracker] = dice_mat_temp
    softmax_matrix_tracker[:,:,:,:,counter_tracker] = softmax_matrix
    seg_matrix_tracker[:,:,:,counter_tracker] = seg_matrix

    path_to_save_figures = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/Figures/"

    # no corrections
    plt.figure(figsize=(25,60))
    plt.subplot(241).title.set_text("Original")
    plt.imshow(original_image)
    plt.subplot(242).title.set_text("CV Map")
    plt.imshow(cv_map_temp)
    plt.colorbar()
    plt.subplot(243).title.set_text("GT mask")
    plt.imshow(GT_mask)
    plt.subplot(244).title.set_text("CV Map x GT mask")
    plt.imshow(GT_cv)
    plt.colorbar()
    plt.subplot(245).title.set_text("Original")
    plt.imshow(original_image)
    plt.subplot(246).title.set_text("Entropy Map")
    plt.imshow(ent_map_temp)
    plt.colorbar()
    plt.subplot(247).title.set_text("GT mask")
    plt.imshow(GT_mask)
    plt.subplot(248).title.set_text("Entropy Map x GT mask")
    plt.imshow(GT_ent)
    plt.colorbar()
    plt.suptitle("Immagine: "+GT_seg_name,fontsize=16)
    plt.show()
    plt.savefig(path_to_save_figures+"Figures_"+GT_seg_name)
    plt.close()

    # Entropia normalizzata
    norm_ent_map = maxminscal(ent_map_temp)
    norm_GT_ent = GT_mask * norm_ent_map
    plt.figure(figsize=(25,60))
    plt.subplot(241).title.set_text("Original")
    plt.imshow(original_image)
    plt.subplot(242).title.set_text("CV Map")
    plt.imshow(cv_map_temp)
    plt.colorbar()
    plt.subplot(243).title.set_text("GT mask")
    plt.imshow(GT_mask)
    plt.subplot(244).title.set_text("CV Map x GT mask")
    plt.imshow(GT_cv)
    plt.colorbar()
    plt.subplot(245).title.set_text("Original")
    plt.imshow(original_image)
    plt.subplot(246).title.set_text("Entropy Map")
    plt.imshow(norm_ent_map)
    plt.colorbar()
    plt.subplot(247).title.set_text("GT mask")
    plt.imshow(GT_mask)
    plt.subplot(248).title.set_text("Entropy Map x GT mask")
    plt.imshow(norm_GT_ent)
    plt.colorbar()
    plt.suptitle("Immagine: "+GT_seg_name,fontsize=16)
    plt.show()
    plt.savefig(path_to_save_figures+"Figures_norm_ent_"+GT_seg_name)
    plt.close()
    
    # Entropia tolto il minimo
    min_ent = ent_map_temp - np.min(ent_map_temp)
    min_GT_ent = GT_mask * min_ent
    plt.figure(figsize=(25,60))
    plt.subplot(241).title.set_text("Original")
    plt.imshow(original_image)
    plt.subplot(242).title.set_text("CV Map")
    plt.imshow(cv_map_temp)
    plt.colorbar()
    plt.subplot(243).title.set_text("GT mask")
    plt.imshow(GT_mask)
    plt.subplot(244).title.set_text("CV Map x GT mask")
    plt.imshow(GT_cv)
    plt.colorbar()
    plt.subplot(245).title.set_text("Original")
    plt.imshow(original_image)
    plt.subplot(246).title.set_text("Entropy Map")
    plt.imshow(min_ent)
    plt.colorbar()
    plt.subplot(247).title.set_text("GT mask")
    plt.imshow(GT_mask)
    plt.subplot(248).title.set_text("Entropy Map x GT mask")
    plt.imshow(min_GT_ent)
    plt.colorbar()
    plt.suptitle("Immagine: "+GT_seg_name,fontsize=16)
    plt.show()
    plt.savefig(path_to_save_figures+"Figures_menomin_ent_"+GT_seg_name)
    plt.close()

    plt.figure()
    plt.imshow(dice_mat_temp)
    plt.title("Dice mat img: "+GT_seg_name, fontsize=16)
    plt.colorbar()
    plt.show()
    plt.savefig(path_to_save_figures+"Dice_mat_figures/"+GT_seg_name)
    plt.savefig(path_to_save_figures+GT_seg_name[:-4]+"_dice.png")
    plt.close()
    
    counter_tracker += 1

dice_mat_tracker[dice_mat_tracker<0] = np.nan
#%%

#%% Saving of dataframes
metrics_df_path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"

index_array = np.arange(1,51)
metrics_dict = {'index':index_array,
                'image_tracker':image_tracker, 
                # 'x_metrica_cv_map':x_metrica_cv_map,
                'x_metrica_cv_map_sum':x_metrica_cv_map_sum,
                # 'x_metrica_ent_map':x_metrica_ent_map,
                'x_metrica_ent_map_sum':x_metrica_ent_map_sum,
                'x_metrica_dice_multi':x_metrica_dice_multi,
                'mean_dice_for_image':mean_dice_for_image}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(metrics_df_path+"metrics_v6.csv",index=False)

#%%

# plt.figure(figsize=(40,10))
# plt.subplot(141).title.set_text("Metrica basata su CV")
# plt.plot(index_array,x_metrica_cv_map,'b')
# plt.subplot(142).title.set_text("Metrica basata su Entropy")
# plt.plot(index_array,x_metrica_ent_map,'tab:orange' )
# plt.subplot(143).title.set_text("Metrica basata su matrice dei dice")
# plt.plot(index_array,x_metrica_dice_multi,'g')
# plt.subplot(144).title.set_text("Overview metriche (normalizzate)")
# plt.plot(index_array,maxminscal(x_metrica_cv_map),'b')
# plt.plot(index_array,maxminscal(x_metrica_ent_map),'tab:orange')
# plt.plot(index_array,maxminscal(x_metrica_dice_multi),'g')
# plt.legend(['metrica cv_map','metrica ent_map','metrica_matrice_dice'],loc='upper right')
# plt.suptitle("Metriche per immagine",fontsize=16)
# plt.show()
# plt.savefig(metrics_df_path+"Metriche per immagine.png")

# plt.figure(figsize=(40,10))
# plt.subplot(141).title.set_text("Metrica basata su CV")
# plt.scatter(mean_dice_for_image,x_metrica_cv_map,c='b')
# plt.xlabel("Mean Dice (su 20)")
# plt.ylabel("Metrica basata su CV")
# plt.subplot(142).title.set_text("Metrica basata su Entropy")
# plt.scatter(mean_dice_for_image,x_metrica_ent_map,c='tab:orange')
# plt.xlabel("Mean Dice (su 20)")
# plt.ylabel("Metrica basata su Entropy")
# plt.subplot(143).title.set_text("Metrica basata su matrice dei dice")
# plt.scatter(mean_dice_for_image,x_metrica_dice_multi,c='g')
# plt.xlabel("Mean Dice (su 20)")
# plt.ylabel("Metrica basata su matrice dei dice")
# plt.subplot(144).title.set_text("Overview metriche (normalizzate)")
# plt.scatter(maxminscal(mean_dice_for_image),maxminscal(x_metrica_cv_map),c='b')
# plt.scatter(maxminscal(mean_dice_for_image),maxminscal(x_metrica_ent_map),c='tab:orange')
# plt.scatter(maxminscal(mean_dice_for_image),maxminscal(x_metrica_dice_multi),c='g')
# plt.xlabel("Mean Dice (su 20)")
# plt.legend(['metrica cv_map','metrica ent_map','metrica_matrice_dice'],loc='upper left')
# plt.suptitle("Scatterplot metriche x dice medio per immagine",fontsize=16)
# plt.show()
# plt.savefig(metrics_df_path+"Scatterplot metriche x dice medio.png")

# plt.figure(figsize=(30,10))
# plt.subplot(131).title.set_text("Confronto CV / Entropy")
# plt.scatter(x_metrica_cv_map,x_metrica_ent_map)
# plt.xlabel("Metrica basata su CV")
# plt.ylabel("Metrica basata su Entropy")
# plt.subplot(132).title.set_text("Confronto CV / Dice Matrix")
# plt.scatter(x_metrica_cv_map,x_metrica_dice_multi)
# plt.xlabel("Metrica basata su CV")
# plt.ylabel("Metrica basata su matrice dei dice")
# plt.subplot(133).title.set_text("Confronto Dice Matrix / Entropy")
# plt.scatter(x_metrica_dice_multi,x_metrica_ent_map)
# plt.xlabel("Metrica basata su matrice dei dice")
# plt.ylabel("Metrica basata su Entropy")
# plt.suptitle("Confronto fra le tre metriche")
# plt.show()
# plt.savefig(metrics_df_path+"Confronto tra le metriche.png")

#%%

plt.figure(figsize=(40,10))
plt.subplot(141).title.set_text("Metrica basata su CV")
plt.plot(index_array,x_metrica_cv_map_sum,'b')
plt.subplot(142).title.set_text("Metrica basata su Entropy")
plt.plot(index_array,x_metrica_ent_map_sum,'tab:orange' )
plt.subplot(143).title.set_text("Metrica basata su matrice dei dice")
plt.plot(index_array,x_metrica_dice_multi,'g')
plt.subplot(144).title.set_text("Overview metriche (normalizzate)")
plt.plot(index_array,maxminscal(x_metrica_cv_map_sum),'b')
plt.plot(index_array,maxminscal(x_metrica_ent_map_sum),'tab:orange')
plt.plot(index_array,maxminscal(x_metrica_dice_multi),'g')
plt.legend(['metrica cv_map','metrica ent_map','metrica_matrice_dice'],loc='upper right')
plt.suptitle("Metriche per immagine",fontsize=16)
plt.show()
plt.savefig(metrics_df_path+"Metriche per immagine _ sum.png")

plt.figure(figsize=(40,10))
plt.subplot(141).title.set_text("Metrica basata su CV")
plt.scatter(mean_dice_for_image,x_metrica_cv_map_sum,c='b')
plt.xlabel("Mean Dice (su 20)")
plt.ylabel("Metrica basata su CV")
plt.subplot(142).title.set_text("Metrica basata su Entropy")
plt.scatter(mean_dice_for_image,x_metrica_ent_map_sum,c='tab:orange')
plt.xlabel("Mean Dice (su 20)")
plt.ylabel("Metrica basata su Entropy")
plt.subplot(143).title.set_text("Metrica basata su matrice dei dice")
plt.scatter(mean_dice_for_image,x_metrica_dice_multi,c='g')
plt.xlabel("Mean Dice (su 20)")
plt.ylabel("Metrica basata su matrice dei dice")
plt.subplot(144).title.set_text("Overview metriche (normalizzate)")
plt.scatter(maxminscal(mean_dice_for_image),maxminscal(x_metrica_cv_map_sum),c='b')
plt.scatter(maxminscal(mean_dice_for_image),maxminscal(x_metrica_ent_map_sum),c='tab:orange')
plt.scatter(maxminscal(mean_dice_for_image),maxminscal(x_metrica_dice_multi),c='g')
plt.xlabel("Mean Dice (su 20)")
plt.legend(['metrica cv_map','metrica ent_map','metrica_matrice_dice'],loc='upper left')
plt.suptitle("Scatterplot metriche x dice medio per immagine",fontsize=16)
plt.show()
plt.savefig(metrics_df_path+"Scatterplot metriche x dice medio _ sum.png")

plt.figure(figsize=(30,10))
plt.subplot(131).title.set_text("Confronto CV / Entropy")
plt.scatter(x_metrica_cv_map_sum,x_metrica_ent_map_sum)
plt.xlabel("Metrica basata su CV")
plt.ylabel("Metrica basata su Entropy")
plt.subplot(132).title.set_text("Confronto CV / Dice Matrix")
plt.scatter(x_metrica_cv_map_sum,x_metrica_dice_multi)
plt.xlabel("Metrica basata su CV")
plt.ylabel("Metrica basata su matrice dei dice")
plt.subplot(133).title.set_text("Confronto Dice Matrix / Entropy")
plt.scatter(x_metrica_dice_multi,x_metrica_ent_map_sum)
plt.xlabel("Metrica basata su matrice dei dice")
plt.ylabel("Metrica basata su Entropy")
plt.suptitle("Confronto fra le tre metriche")
plt.show()
plt.savefig(metrics_df_path+"Confronto tra le metriche _ sum.png")

#%%
mega_matrici_path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
np.save(mega_matrici_path+"dice_mat_tracker",dice_mat_tracker)              # dice_mat_tracker is [20,20,50]
np.save(mega_matrici_path+"softmax_matrix_tracker",softmax_matrix_tracker)  # softmax_matrix_tracker is [416,416,2,20,50]
np.save(mega_matrici_path+"seg_matrix_tracker",seg_matrix_tracker)          # seg_matrix_tracker is [416,416,20,50]