# piccolo appunto: entropy sum e entropy mean, cosi come cv sum e cv mean, sono matematicamente correlati
# quindi Ã¨ inutile avere sia sum che mean

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
        softmax_matrix[:,:,counter] = np.copy(st)
        counter += 1
    
    entropy_map = m.entropy(softmax_matrix)
    entropy_sum = m.entropy_sum(entropy_map)
    # entropy_mean = m.entropy_mean(entropy_map)
    
    cv_map = m.cv(softmax_matrix)
    cv_sum = m.cv_sum(cv_map)
    # cv_mean = m.cv_mean(cv_map)
    
    # seg_GT = cv2.imread(GT_seg_dir + GT_seg_name)
    # seg_MC = cv2.imread(seg_MC_path)
    pngseg_GT = Img.open(GT_seg_dir + GT_seg_name)
    pngseg_MC = Img.open(seg_MC_path)
    seg_GT = np.array(pngseg_GT)/255
    seg_MC = np.array(pngseg_MC)/255
    
    DICE = m.dice(seg_GT,seg_MC)
    
    DATASET[i,0] = entropy_sum

    DATASET[i,1] = cv_sum

    DATASET[i,2] = DICE
    
    i += 1

#%% DATAFRAME AND DATASET SPLIT
path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
df = pd.DataFrame(DATASET)
df = df.rename(columns={0:'ent_sum',1:'cv_sum',2:'dice'})
df.to_csv(path+"DATASET_v4.csv",index=False)

#%%
DATA = DATASET
ent = DATA[:,0]
cv = DATA[:,1]
dice = DATA[:,2]

#%% m for mean, st for standard deviation
ment = np.mean(ent)
stent = np.std(ent)
mcv = np.mean(cv)
stcv = np.std(cv)
mdice = np.mean(dice)
stdice = np.std(dice)

idxentM = np.where(ent>ment+3*stent)
idxentm = np.where(ent<ment-3*stent)
idxcvM = np.where(cv>mcv+3*stcv)
idxcvm = np.where(cv<mcv-3*stcv)
idxdiceM = np.where(dice>mdice+3*stdice)
idxdicem = np.where(dice<mdice-3*stdice)

idx_list = np.concatenate((idxentM,idxentm,idxcvM,idxcvm,idxdiceM,idxdicem),1)
idx_toremove = np.unique(idx_list)

idx = np.setdiff1d(np.arange(50),idx_toremove)

#%% c for clean
cent = ent[idx]
ccv = cv[idx]
cdice = dice[idx]

CLEAN_DATASET = np.zeros((len(cent),3))
cd_path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
CLEAN_DATASET[:,0] = cent
CLEAN_DATASET[:,1] = ccv
CLEAN_DATASET[:,2] = cdice
df = pd.DataFrame(CLEAN_DATASET)
df = df.rename(columns={0:'ent_sum',1:'cv_sum',2:'dice'})
df.to_csv(cd_path+"CLEAN_DATASET_v4.csv",index=False)

mcent = np.mean(cent)
mccv = np.mean(ccv)
mcdice = np.mean(cdice)

#%% n for normalized
entM = max(cent)
entm = min(cent)
cvM = max(ccv)
cvm = min(ccv)
diceM = max(cdice)
dicem = min(cdice)

nent = (cent-entm)/(entM-entm)
ncv = (ccv-cvm)/(cvM-cvm)
ndice = (cdice-dicem)/(diceM-dicem)

# plt.plot(nent)
# plt.plot(ncv)
# plt.plot(ndice)

#%%
perc_split = 0.8
tot = len(cent)
idx_divide = np.round(perc_split*tot)
idx_divide = np.uint(idx_divide)

trent = cent[:idx_divide]
trcv = ccv[:idx_divide]
trdice = cdice[:idx_divide]

testent = cent[idx_divide:]
testcv = ccv[idx_divide:]
testdice = cdice[idx_divide:]

#%% Linear Regression
slope_ent, intercept_ent, r_ent, p_ent, _ = stats.linregress(trent,trdice)
slope_cv, intercept_cv, r_cv, p_cv, _ = stats.linregress(trcv,trdice)

line_ent = lin_regr(slope_ent,trent,intercept_ent)
line_cv = lin_regr(slope_cv,trcv,intercept_cv)

#%% Multiple Linear Regression
X_tr = np.transpose([trent,trcv])
X_test = np.transpose([testent,testcv])

mult_lin_regr = linear_model.LinearRegression()
mult_lin_regr.fit(X_tr,trdice)

#%% TEST
y_pred_ent_tr = np.copy(line_ent)
y_pred_cv_tr = np.copy(line_cv)

y_pred_ent_test = lin_regr(slope_ent,testent,intercept_ent)
y_pred_cv_test = lin_regr(slope_cv,testcv,intercept_cv)

y_mlr_tr = mult_lin_regr.predict(X_tr)
y_mlr_test = mult_lin_regr.predict(X_test)

#%%
# Calculate mean squared error and mean absolute error for entity_train
mse_ent_train = mean_squared_error(trdice, y_pred_ent_tr)
mae_ent_train = mean_absolute_error(trdice, y_pred_ent_tr)

# Calculate mean squared error and mean absolute error for entity_test
mse_ent_test = mean_squared_error(testdice, y_pred_ent_test)
mae_ent_test = mean_absolute_error(testdice, y_pred_ent_test)

# Calculate mean squared error and mean absolute error for cv_train
mse_cv_train = mean_squared_error(trdice, y_pred_cv_tr)
mae_cv_train = mean_absolute_error(trdice, y_pred_cv_tr)

# Calculate mean squared error and mean absolute error for cv_test
mse_cv_test = mean_squared_error(testdice, y_pred_cv_test)
mae_cv_test = mean_absolute_error(testdice, y_pred_cv_test)

# Calculate mean squared error and mean absolute error for mlr_train
mse_mlr_train = mean_squared_error(trdice, y_mlr_tr)
mae_mlr_train = mean_absolute_error(trdice, y_mlr_tr)

# Calculate mean squared error and mean absolute error for mlr_test
mse_mlr_test = mean_squared_error(testdice, y_mlr_test)
mae_mlr_test = mean_absolute_error(testdice, y_mlr_test)

#%%
plt.figure(figsize=(14,7))

plt.subplot(211)
plt.plot(y_pred_cv_tr, 'g', label='cv regression')
plt.plot(y_pred_ent_tr, 'b', label='entropy regression')
plt.plot(y_mlr_tr, 'c', label='multilinear regression')
plt.plot(trdice, 'r', label='dice')
plt.legend(loc='lower right')
plt.title("Training Set Predictions")

plt.subplot(212)
plt.plot(y_pred_cv_test, 'g', label='cv regression')
plt.plot(y_pred_ent_test, 'b', label='entropy regression')
plt.plot(y_mlr_test, 'c', label='multilinear regression')
plt.plot(testdice, 'r', label='dice')
plt.legend(loc='lower right')
plt.title("Test Set Predictions")

plt.show()

#%%
plt.figure(figsize=(17,13))
plt.subplot(411)
plt.plot(nent,'b')
plt.title("Normalized entropy")
plt.subplot(412)
plt.plot(ncv, 'g')
plt.title("Normalized cv")
plt.subplot(413)
plt.plot(ndice, 'r')
plt.title("Normalized dice")
plt.subplot(414)
plt.plot(nent,'b', label='normalized entropy')
plt.plot(ncv, 'g', label='normalized cv')
plt.plot(ndice, 'r', label='normalized dice')
plt.title("Normalized metrics")
plt.legend(loc='lower right')
plt.show()
