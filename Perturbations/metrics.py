import numpy as np
from scipy.stats import entropy

#Entropy of a softmax
def softmax_entropy(softmax):
    ent = -np.sum(softmax*np.log(softmax)) #log2 non si può fare per 0, quindi fare per softmax>0
    # ent = entropy(softmax)
    return ent

#Cross Entropy between two softmax
def cross_entropy(softmax1,softmax2):
    ce = -np.sum(softmax1*np.log(softmax2))
    return ce

#RMSE
def rmse(softmax1,softmax2):
    mse = np.mean((softmax1-softmax2)**2)
    rmse_val = np.sqrt(mse)
    return rmse_val

#Kullback-Leibler Divergence / Relative Entropy
def KL_divergence(softmax1,softmax2):
    #Important: KL Divergence is not simmetric, so it could be
    #Calculated twice, or the original must be put first
    kl_div = entropy(softmax1,softmax2)
    return kl_div

def dice(seg1, seg2):
    dice_coeff = -1
    if np.sum(seg1)+np.sum(seg2) > 0:
        intersection = np.sum(seg1 * seg2)
        dice_coeff = (2.0 * intersection) / (np.sum(seg1) + np.sum(seg2))  
    return dice_coeff