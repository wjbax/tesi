# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:31:00 2024

@author: willy
"""

#%% IMPORT

import numpy as np
import matplotlib.pyplot as plt
import cv2

#%% FUNCTIONS
#%%% Metrics
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

#%%% Mask manipulation

def mask_splitter(mask):
    mask_C1 = np.copy(mask)
    mask_C2 = np.copy(mask)

    mask_C1[mask > 0.8] = 0
    mask_C1[mask_C1 > 0] = 1
    mask_C2[mask < 0.8] = 0
    mask_C2[mask_C2 > 0] = 1
    return mask_C1, mask_C2


#%% PATHS & OPEN IMG

path_2c = 'D:/DATASET_Tesi_marzo2024/Liver HE steatosis/DATASET_2classes/test/manual/1004289_35.png'
path_3c = 'D:/DATASET_Tesi_marzo2024/Liver HE steatosis/DATASET_3classes/test/manual/1004289_35.png'

mask_2c = cv2.imread(path_2c, cv2.IMREAD_GRAYSCALE).astype(bool)
mask_3c = cv2.imread(path_3c, cv2.IMREAD_GRAYSCALE)/255
mask_3c_ext, mask_3c_int = mask_splitter(mask_3c)
# mask_3c_ext = mask_3c_ext.astype(bool)
# mask_3c_int = mask_3c_int.astype(bool)

#%% DILATE LOOP

max_dice = dice(mask_2c,mask_3c_int.astype(bool))
for i in range(5):
    kernel = np.ones((i,i),np.uint8)
    mask_3c_int_dil = cv2.dilate(mask_3c_int,kernel).astype(bool)
    dice_i = dice(mask_3c_int_dil,mask_2c)
    if dice_i > max_dice: 
        max_dice = dice_i
        print("Se raggio è " + str(i) + " dice è " + str(dice_i))