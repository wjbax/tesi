# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:44:59 2024

@author: willy
"""

def mask_generator_for_calcoli(SI_mask_C1,SI_mask_C2,uncert_map_C1,uncert_map_C2,Th,softmax_matrix): #da fare una volta per classe
    
    # mean_softmax = np.mean(softmax_matrix,axis=-1)    
    # mask_avg = np.argmax(mean_softmax,axis=-1)/2
    # mask_union da calcolare
    # mask_background = (~mask_union)
    
    mask_avg_C1,mask_avg_C2 = mask_splitter(mask_avg)
    mask_avg_C1=mask_avg_C1.astype(bool)
    mask_avg_C2=mask_avg_C2.astype(bool)
    
    mask_uncert_C1 = uncert_map_C1 > Th   #(0.1, 0.2, 0.3, ...)
    mask_cert_C1 = (~mask_uncert_C1)
    mask_cert_C1[mask_background] = False
    
    mask_uncert_C2 = uncert_map_C2 > Th   #(0.1, 0.2, 0.3, ...)
    mask_cert_C2 = (~mask_uncert_C2) 
    
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