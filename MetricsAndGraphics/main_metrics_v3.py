# piccolo appunto: entropy sum e entropy mean, cosi come cv sum e cv mean, sono matematicamente correlati
# quindi è inutile avere sia sum che mean

#%% import
import os
import numpy as np
import pandas as pd
import metrics_v2 as m
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score
from sklearn import linear_model

#%% functions
def lin_regr(slope,x,intercept):
    return slope*x+intercept

#%% PATH INITIALIZATION
SOFTMAX_DIM = [416,416]
GT_seg_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/DATASET/test/manual/"
softmax_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/softmax/"
seg_MC_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS/test/mask/"

#%%
DATASET = np.zeros([50,5])
i = 0
for GT_seg_name in tqdm(os.listdir(GT_seg_dir)):
    softmax_path = softmax_dir + GT_seg_name + "/"
    seg_MC_path = seg_MC_dir + GT_seg_name
    MC_softmax_list = os.listdir(softmax_path)
    softmax_matrix = np.zeros([SOFTMAX_DIM[0],SOFTMAX_DIM[1],len(MC_softmax_list)])
    counter = 0
    for name in MC_softmax_list:
        st = np.float32((np.load(softmax_path + name)['softmax'])[:,:,1])
        softmax_matrix[:,:,counter] = np.copy(st)
        counter += 1
    
    entropy_map = m.entropy(softmax_matrix)
    entropy_sum = m.entropy_sum(entropy_map)
    entropy_mean = m.entropy_mean(entropy_map)
    
    cv_map = m.cv(softmax_matrix)
    cv_sum = m.cv_sum(cv_map)
    cv_mean = m.cv_mean(cv_map)
    
    seg_GT = cv2.imread(GT_seg_dir + GT_seg_name)
    seg_MC = cv2.imread(seg_MC_path)
    
    DICE = m.dice(seg_GT,seg_MC)
    
    DATASET[i,0] = entropy_sum
    DATASET[i,1] = entropy_mean
    DATASET[i,2] = cv_sum
    DATASET[i,3] = cv_mean
    DATASET[i,4] = DICE
    
    i += 1

#%% DATAFRAME AND DATASET SPLIT
path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
df = pd.DataFrame(DATASET)
df = df.rename(columns={0:'esum',1:'emean',2:'cvsum',3:'cvmean',4:'dice'})
df.to_csv(path+"DATASET_v2.csv",index=False)

TRAIN = DATASET[0:40,:]
TEST = DATASET[40:,:]

#%% NO DICE 0 VALUES
ent_sum = TRAIN[:,0]
ent_mean = TRAIN[:,1]
cv_s = TRAIN[:,2]
cv_m = TRAIN[:,3]
y = TRAIN[:,4]

x1 = ent_sum[y>0]
x2 = ent_mean[y>0]
x3 = cv_s[y>0]
x4 = cv_m[y>0]
y = y[y>0]

X = np.transpose(np.array([x1,x2,x3,x4]))
data = np.transpose(np.array([x1,x2,x3,x4,y]))
data = pd.DataFrame(data).rename(columns={0:'x1',1:'x2',2:'x3',3: 'x4',4:'y'})
#%% DATA ANALYSIS
#%%% Linear Regression (4 variables, 4 regr)
slope1, intercept1, r_x1, p1, _ = stats.linregress(x1,y)
slope2, intercept2, r_x2, p2, _ = stats.linregress(x2,y)
slope3, intercept3, r_x3, p3, _ = stats.linregress(x3,y)
slope4, intercept4, r_x4, p4, _ = stats.linregress(x4,y)

line1 = lin_regr(slope1,x1,intercept1)
line2 = lin_regr(slope2,x2,intercept2)
line3 = lin_regr(slope3,x3,intercept3)
line4 = lin_regr(slope4,x4,intercept4)

#%%% Multiple Linear Regression (4 variables, 1 m_regr)
mult_lin_regr = linear_model.LinearRegression()
mult_lin_regr.fit(X,y)

#%%% Support Vector Regression
# Feature scaling - it's a good practice for SVR
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)

# Creating and training the SVR model
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # Using a radial basis function (RBF) kernel
svr.fit(X_scaled, y)

# Predicting on the test set
y_pred = svr.predict(X_scaled)

# Evaluating the model
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")

#%%
sp_1 = stats.spearmanr(x1,y)
sp_2 = stats.spearmanr(x2,y)
sp_3 = stats.spearmanr(x3,y)
sp_4 = stats.spearmanr(x4,y)

print(sp_1[0])
print(sp_2[0])
print(sp_3[0])
print(sp_4[0])

pc_1 = stats.pearsonr(x1, y)
pc_2 = stats.pearsonr(x2, y)
pc_3 = stats.pearsonr(x3, y)
pc_4 = stats.pearsonr(x4, y)

print(pc_1[0])
print(pc_2[0])
print(pc_3[0])
print(pc_4[0])

pc_12 = stats.pearsonr(x1,x2)
print(pc_12[0])
sp_12 = stats.spearmanr(x1,x2)
print(sp_12[0])

#%% 
a = DATASET[:,0]
b = DATASET[:,4]
c = DATASET[:,2]

MA = max(a)
MB = max(b)
MC = max(c)

ma = min(a)
mb = min(b)
mc = min(c)

ap = (a-ma)/(MA-ma)
bp = (b-mb)/(MB-mb)
cp = (c-mc)/(MC-mc)

plt.figure()
plt.plot(a)
# plt.plot(b)
# plt.plot(c)
plt.show()