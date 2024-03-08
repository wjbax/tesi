# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# import matplotlib.pyplot as plt

# dataset_path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/CLEAN_DATASET.csv"
# df = pd.read_csv(dataset_path)

# fig = px.density_heatmap(df, x="ent_sum", y="dice")
# fig.show(renderer="png")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
bins = 20
dataset_path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/CLEAN_DATASET_v4.csv"
df = pd.read_csv(dataset_path)

x_ent = np.array(df["ent_sum"])
x_cv = np.array(df["cv_sum"])
y_dice = np.array(df["dice"])

hist_values_ent, x_edges_ent, y_edges_ent = np.histogram2d(x_ent, y_dice, bins=bins)
hist_values_cv, x_edges_cv, y_edges_cv = np.histogram2d(x_cv, y_dice, bins=bins)

#%%
y_redline_ent = np.zeros(bins+1)
x_redline_ent = np.zeros(bins+1)
for i in range(0,len(x_edges_ent)):
    m = x_edges_ent[i-1]
    M = x_edges_ent[i]
    x_temp = x_ent[np.where(x_ent > m)]
    x_temp = x_temp[np.where(x_temp <= M)]
    y_temp = y_dice[np.where(x_ent > m)]
    y_temp = y_temp[np.where(x_temp <= M)]
    mean_y = np.mean(y_temp)
    x_redline_ent[i]=(m+M)/2
    y_redline_ent[i]=mean_y
    print(str(i) + " " + str(mean_y))
    
# mean_y_line = np.nanmean(y_redline_ent)
# y_redline_ent[np.isnan(y_redline_ent)] = 0

y_redline_cv = np.zeros(bins+1)
x_redline_cv = np.zeros(bins+1)
for i in range(0,len(x_edges_cv)):
    m = x_edges_cv[i-1]
    M = x_edges_cv[i]
    x_temp = x_cv[np.where(x_cv > m)]
    x_temp = x_temp[np.where(x_temp <= M)]
    y_temp = y_dice[np.where(x_cv > m)]
    y_temp = y_temp[np.where(x_temp <= M)]
    mean_y = np.mean(y_temp)
    x_redline_cv[i]=(m+M)/2
    y_redline_cv[i]=mean_y
    print(str(i) + " " + str(mean_y))

# mean_y_line = np.nanmean(y_redline_cv)
# y_redline_cv[np.isnan(y_redline_cv)] = 0

#%%

plt.figure(figsize=(6,12))
plt.subplot(211)
plt.hist2d(x_ent,y_dice,bins=bins)
plt.plot(x_redline_ent,y_redline_ent, color="red")
plt.colorbar()
plt.subplot(212)
plt.hist2d(x_cv,y_dice,bins=bins)
plt.plot(x_redline_cv,y_redline_cv, color="r")
plt.colorbar()
plt.show()

#%%

# dataset_path = "C:/Users/willy/Desktop/Tesi_v2/tesi/data_saves/DATASET_v4.csv"
# df = pd.read_csv(dataset_path)

# x_ent = np.array(df["ent_sum"])
# x_cv = np.array(df["cv_sum"])
# y_dice = np.array(df["dice"])
# plt.figure(figsize=(6,12))
# plt.subplot(211)
# plt.hist2d(x_ent,y_dice)
# plt.subplot(212)
# plt.hist2d(x_cv,y_dice)
# plt.show()
