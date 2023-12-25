#%% import
import numpy as np
#%% Entropy

# solo uno strato della softmax può essere inserito in questa funzione
def entropy(softmax_matrix):
    shape = np.shape(softmax_matrix)
    entropy_map = np.zeros([shape[0],shape[1]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            ij_vector = softmax_matrix[i,j,:]
            sum_ij_vector = sum(ij_vector)
            ent = np.zeros(len(ij_vector))
            for k in range(len(ij_vector)):
                pt_k = (softmax_matrix[i,j,k])/sum_ij_vector
                if pt_k == 0:
                    ent[k] = np.nan
                else:
                    log_k = np.log2(pt_k)
                    ent[k] = pt_k*log_k
            entropy_map[i,j] = -np.nansum(ent)
    return entropy_map

def entropy_sum(entropy_map):
    return np.nansum(entropy_map)

def entropy_mean(entropy_map):
    if np.nansum(entropy_map) == 0:
        return 0
    return np.nanmean(entropy_map)

#%% Dice
def dice(seg1, seg2):
    dice_coeff = -1
    if np.sum(seg1)+np.sum(seg2) > 0:
        intersection = np.sum(seg1 * seg2)
        dice_coeff = (2.0 * intersection) / (np.sum(seg1) + np.sum(seg2))  
    return dice_coeff

#%%
