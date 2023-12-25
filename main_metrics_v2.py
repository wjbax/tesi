#%% import
import os
import numpy as np
import pandas as pd
import metrics_v2 as m
import matplotlib.pyplot as plt
import time
import cv2
from tqdm import tqdm
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

#%% dataset
SOFTMAX_DIM = [416,416]
GT_seg_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/DATASET/test/manual/"
softmax_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS_MC/test/softmax/"
seg_MC_dir = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver HE Steatosis (TEMP)/Liver HE Steatosis (TEMP)/k-net+swin/TEST_2classes/RESULTS/test/mask/"
TABELLA = np.zeros([50,3])
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
    
    # TABELLA[i,0] = entropy_sum
    # TABELLA[i,1] = entropy_mean
    # TABELLA[i,2] = DICE
    
    i += 1
#%%
path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
df = pd.DataFrame(TABELLA)
df = df.rename(columns={0:'ent_sum',1:'ent_mean',2:'dice'})
df.to_csv(path+"TABELLA_v1.csv",index=False)

#%%
ent_sum = TABELLA[:,0]
ent_mean = TABELLA[:,1]
y = TABELLA[:,2]

x1 = ent_sum[y>0]
x2 = ent_mean[y>0]
y = y[y>0]

x1a = x1[x1>747860]
y1a = y[x1>747860]
x2a = x2[x2>0.0005]
y2a = y[x2>0.0005]

X = np.transpose(np.array([x1,x2]))
#%% LINEAR REGRESSION
slope1, intercept1, r1, p1, std_err1 = stats.linregress(x1,y)
slope2, intercept2, rb, p2, std_err2 = stats.linregress(x2,y)

line1 = slope1*x1+intercept1
line2 = slope2*x2+intercept2

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title("ESum vs DICE: r = " + str(r1))
plt.scatter(x1,y)
plt.plot(x1,line1)
plt.subplot(122)
plt.title("EMean vs DICE: r = " + str(rb))
plt.scatter(x2,y)
plt.plot(x2,line2)
plt.show()

#%%
# slope1a, intercept1a, r1a, p1a, std_err1a = stats.linregress(x1a,y1a)
# slope2a, intercept2a, r2a, p2a, std_err2a = stats.linregress(x2a,y2a)

# line1a = slope1a*x1a+intercept1a
# line2a = slope2a*x2a+intercept2a

# plt.figure(figsize=(10,5))
# plt.subplot(121)
# plt.title("ESum vs DICE: r = " + str(r1a))
# plt.scatter(x1a,y1a)
# plt.plot(x1a,line1a)
# plt.subplot(122)
# plt.title("EMean vs DICE: r = " + str(r2a))
# plt.scatter(x2a,y2a)
# plt.plot(x2a,line2a)
# plt.show()

#%% linear regression multivariate
regr = linear_model.LinearRegression()
regr.fit(X,y)
mv_regr = regr.predict(X)

#%%
r2_Esum = r2_score(y,slope1*x1+intercept1)
r2_Emean = r2_score(y,slope2*x2+intercept2)
r2_regr = r2_score(y,mv_regr)

print(r2_Esum)
print(r2_Emean)
print(r2_regr)

#%%
X_clust = TABELLA[TABELLA[:,2]>0]
linkage_data = linkage(X_clust, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()

#%%
hierarchical_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(X_clust)

plt.scatter(x2, y, c=labels)
plt.show()
