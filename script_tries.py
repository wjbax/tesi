import numpy as np
import cv2
import transformations as t
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

image_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/DATASET/train/train/image/1001051_19.png"
seg_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/DATASET/train/train/manual/1001051_19.png"
softmax_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/Liver steatosis HE/k-net+swin/TEST_2classes/RESULTS/train/softmax/1001051_19.npz"

#%%
image = cv2.imread(image_path)
image = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB)
image = image.astype(np.float32)

seg = cv2.imread(seg_path)
seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
seg = np.float32(seg)

b = np.load(softmax_path)['softmax']
softmax = np.float32(b[:,:,1])

original_shape = np.shape(image)

#%% scaling
[new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=t.scaling(1,image,seg,softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
[ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = t.scaling(0,new_image,new_seg,new_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)

#%% rotation
[new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=t.rotation(1,image,seg,softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
[ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = t.rotation(0,new_image,new_seg,new_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)

#%% gaussianblur
[new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=t.gaussblur(1,image,seg,softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
[ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = t.gaussblur(0,new_image,new_seg,new_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)

#%% hsv pert
[new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=t.hsv_pert(1,image,seg,softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
[ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = t.hsv_pert(0,new_image,new_seg,new_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)

#%% hor mirr
[new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=t.hor_mirr(1,image,seg,softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
[ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = t.hor_mirr(0,new_image,new_seg,new_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)

#%% ver mirr
[new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=t.vert_mirr(1,image,seg,softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
[ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = t.vert_mirr(0,new_image,new_seg,new_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)

#%%
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(image/255)
plt.subplot(1,3,2)
plt.imshow(new_image/255)
plt.subplot(1,3,3)
plt.imshow(ric_image/255)

print((image-ric_image).sum())
plt.show()