#%% import
import os
import numpy as np
import pandas as pd
import metrics_v2 as m
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

#%% functions
def lin_regr(slope,x,intercept):
    return slope*x+intercept

#%% PATH INITIALIZATION
SOFTMAX_DIM = [416,416]
GT_seg_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/DATASET/test/manual/"
softmax_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/softmax/"
seg_MC_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS/test/mask/"

#%%
DATASET = np.zeros([50,3])
i = 0
for GT_seg_name in tqdm(os.listdir(GT_seg_dir)):
    softmax_path = softmax_dir + GT_seg_name + "/"
    seg_MC_path = seg_MC_dir + GT_seg_name
    MC_softmax_list = os.listdir(softmax_path)
    softmax_matrix = np.zeros([SOFTMAX_DIM[0],SOFTMAX_DIM[1],len(MC_softmax_list)])
    counter = 0
    for name in MC_softmax_list:
        st = np.float32((np.load(softmax_path + name)['softmax'])[:,:,1])
        softmax_matrix[:,:,counter] = st
        counter += 1
    
    entropy_map = m.entropy(softmax_matrix)
    entropy_sum = m.entropy_sum(entropy_map)
    entropy_mean = m.entropy_mean(entropy_map)
    
    seg_GT = cv2.imread(GT_seg_dir + GT_seg_name)
    seg_MC = cv2.imread(seg_MC_path)
    
    DICE = m.dice(seg_GT,seg_MC)
    
    DATASET[i,0] = entropy_sum
    DATASET[i,1] = entropy_mean
    DATASET[i,2] = DICE
    
    i += 1
#%% DATAFRAME AND DATASET SPLIT
path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
df = pd.DataFrame(DATASET)
df = df.rename(columns={0:'ent_sum',1:'ent_mean',2:'dice'})
df.to_csv(path+"DATASET_v1.csv",index=False)

TRAIN = DATASET[0:40,:]
TEST = DATASET[40:,:]

#%% NO DICE 0 VALUES
ent_sum = TRAIN[:,0]
ent_mean = TRAIN[:,1]
y = TRAIN[:,2]

x1 = ent_sum[y>0]
x2 = ent_mean[y>0]
y = y[y>0]

X = np.transpose(np.array([x1,x2]))
#%% LINEAR REGRESSION FOR x1 AND x2
slope1, intercept1, r1, p1, std_err1 = stats.linregress(x1,y)
slope2, intercept2, rb, p2, std_err2 = stats.linregress(x2,y)

line1 = lin_regr(slope1,x1,intercept1)
line2 = lin_regr(slope2,x2,intercept2)

# plt.figure(figsize=(10,5))
# plt.subplot(121)
# plt.title("ESum vs DICE: r = " + str(r1))
# plt.scatter(x1,y)
# plt.plot(x1,line1)
# plt.subplot(122)
# plt.title("EMean vs DICE: r = " + str(rb))
# plt.scatter(x2,y)
# plt.plot(x2,line2)
# plt.show()

#%% MULTIPLE LINEAR REGRESSION
mult_lin_regr = linear_model.LinearRegression()
mult_lin_regr.fit(X,y)

#%%
r2_Esum = r2_score(y,lin_regr(slope1,x1,intercept1))
r2_Emean = r2_score(y,lin_regr(slope2,x2,intercept2))
r2_regr = r2_score(y,mult_lin_regr.predict(X))

print(r2_Esum)
print(r2_Emean)
print(r2_regr)

#%%
# X_clust = TRAIN[TRAIN[:,2]>0]
# linkage_data = linkage(X_clust, method='ward', metric='euclidean')
# dendrogram(linkage_data)

# plt.show()

# #%%
# hierarchical_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
# labels = hierarchical_cluster.fit_predict(X_clust)

# plt.scatter(x2, y, c=labels)
# plt.show()

#%%
x1_test = TEST[:,0]
x2_test = TEST[:,1]
y_test = TEST[:,2]

y_pred_lr1 = lin_regr(slope1,x1_test,intercept1)
y_pred_lr2 = lin_regr(slope2,x2_test,intercept2)
y_pred_mlr = mult_lin_regr.predict(TEST[:,:2])

#%%
r2_Esum_test = r2_score(y_test,y_pred_lr1)
r2_Emean_test = r2_score(y_test,y_pred_lr2)
r2_regr_test = r2_score(y_test,y_pred_mlr)

print(r2_Esum_test)
print(r2_Emean_test)
print(r2_regr_test)

#%%
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title("ESum vs DICE: r = " + str(r1))
plt.scatter(x1_test,y_test)
plt.plot(x1_test,y_pred_lr1)
plt.subplot(122)
plt.title("EMean vs DICE: r = " + str(rb))
plt.scatter(x2_test,y_test)
plt.plot(x2_test,y_pred_lr2)
plt.show()