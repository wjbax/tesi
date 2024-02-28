# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:29:46 2024

@author: willy
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error 


#%%
def lin_regr(slope,x,intercept):
    return slope*x+intercept


#%%
path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/"
filename = "DATASET_v2.csv"
df = pd.read_csv(path+filename)

temp = np.array(df)

DATA = temp[:,[0,2,4]]

ent = DATA[:,0]
cv = DATA[:,1]
dice = DATA[:,2]

#%%
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

#%%
cent = ent[idx]
ccv = cv[idx]
cdice = dice[idx]

mcent = np.mean(cent)
mccv = np.mean(ccv)
mcdice = np.mean(cdice)

#%%
entM = max(cent)
entm = min(cent)
cvM = max(ccv)
cvm = min(ccv)
diceM = max(cdice)
dicem = min(cdice)

nent = (cent-entm)/(entM-entm)
ncv = (ccv-cvm)/(cvM-cvm)
ndice = (cdice-dicem)/(diceM-dicem)

plt.plot(nent)
plt.plot(ncv)
plt.plot(ndice)

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
print( 
  'mean_squared_error_ent_train : ', mean_squared_error(trdice,y_pred_ent_tr)) 
print( 
  'mean_absolute_error_ent_train : ', mean_absolute_error(trdice,y_pred_ent_tr)) 

print( 
  'mean_squared_error_ent_test : ', mean_squared_error(testdice,y_pred_ent_test)) 
print( 
  'mean_absolute_error_ent_test : ', mean_absolute_error(testdice,y_pred_ent_test))

print( 
  'mean_squared_error_cv_train : ', mean_squared_error(trdice,y_pred_cv_tr)) 
print( 
  'mean_absolute_error_cv_train : ', mean_absolute_error(trdice,y_pred_cv_tr)) 

print( 
  'mean_squared_error_cv_test : ', mean_squared_error(testdice,y_pred_cv_test)) 
print( 
  'mean_absolute_error_cv_test : ', mean_absolute_error(testdice,y_pred_cv_test))

print( 
  'mean_squared_error_mlr_train : ', mean_squared_error(trdice,y_mlr_tr)) 
print( 
  'mean_absolute_error_mlr_train : ', mean_absolute_error(trdice,y_mlr_tr)) 

print( 
  'mean_squared_error_mlr_test : ', mean_squared_error(testdice,y_mlr_test)) 
print( 
  'mean_absolute_error_mlr_test : ', mean_absolute_error(testdice,y_mlr_test))

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
plt.figure(figsize=(10,5))

plt.subplot(211)
plt.plot(y_pred_cv_tr, 'g')
plt.plot(y_pred_ent_tr, 'b')
plt.plot(y_mlr_tr, 'c')
plt.plot(trdice, 'r')

plt.subplot(212)
plt.plot(y_pred_cv_test, 'g')
plt.plot(y_pred_ent_test, 'b')
plt.plot(y_mlr_test, 'c')
plt.plot(testdice, 'r')

plt.show()