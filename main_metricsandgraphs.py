import numpy as np
import metrics as m
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
#%%
DATASET_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/"
names = [d for d in os.listdir(DATASET_path + "RESULTS_MC/test/mask/") if os.path.isdir(os.path.join(DATASET_path + "RESULTS_MC/test/mask/", d))]
n_valutations = 20

DATAFRAME = pd.DataFrame(columns=['Name','Order','entropy','cross_entropy','RMSE','kl_div','dice'])

for name in names:
    softmax_entropy = []
    cross_entropy = []
    RMSE = []
    kl_div = []
    dice = []
    current_name = name[:-4]
    
    orig_softmax_path = DATASET_path + "RESULTS/test/softmax/" + current_name + ".npz"
    orig_seg_path = DATASET_path + "RESULTS/test/mask/" + name
    orig_softmax = np.float32(np.load(orig_softmax_path)['softmax'])
    orig_seg = np.float32(cv2.cvtColor(cv2.imread(orig_seg_path), cv2.COLOR_BGR2GRAY))/255
    
    for counter in range(1,n_valutations+1):
        test_softmax_path = DATASET_path + "RESULTS_MC/test/softmax/" + name + "/" + str(counter) + ".npz"
        test_seg_path = DATASET_path + "RESULTS_MC/test/mask/" + name + "/" + str(counter) + ".png"
        test_softmax = np.float32(np.load(test_softmax_path)['softmax'])
        test_seg = np.float32(cv2.cvtColor(cv2.imread(test_seg_path), cv2.COLOR_BGR2GRAY))/255
        
        softmax_entropy = m.softmax_entropy(test_softmax)
        cross_entropy = m.cross_entropy(orig_softmax, test_softmax)
        RMSE = m.rmse(orig_softmax,test_softmax)
        kl_div = m.KL_divergence(orig_softmax,test_softmax)
        dice = m.dice(orig_seg,test_seg)
        
        row_data_dict = {'Name' : current_name, 
                         'Order' : counter, 
                         'entropy' : softmax_entropy, 
                         'cross_entropy' : cross_entropy,
                         'RMSE':RMSE,
                         'kl_div':kl_div,
                         'dice':dice}
        row_track = pd.DataFrame([row_data_dict])
        DATAFRAME=pd.concat([DATAFRAME,row_track],axis=0)
        
        # del test_softmax
        # del test_seg
        
#%%
selected_columns = ['Name','Order','cross_entropy','RMSE','dice']
selected_data = DATAFRAME[selected_columns].values

#%%
csv_path = "C:/Users/willy/Desktop/Tesi_v2/tesi"
DATAFRAME[selected_columns].to_csv(csv_path+"/DATAFRAME.csv", index=False)






