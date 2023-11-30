import numpy as np
import cv2
import transformations
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random

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

#%%
list_of_transformations = {
    "Rotation" :                transformations.rotation,
    "Vertical mirroring" :      transformations.vert_mirr,
    "Horizontal mirroring" :    transformations.hor_mirr,
    "Scaling" :                 transformations.scaling,
    "Gaussian blurring" :       transformations.gaussblur,
    "HSV perturbations" :       transformations.hsv_pert
}

#%%
img_name = "1001051_19.png"
trasformation_single_image = []
selected_functions = random.sample(list(list_of_transformations.keys()), random.randint(1,4))
if "Scaling" in selected_functions:
    selected_functions.remove("Scaling")
    selected_functions.append("Scaling")
#%%
n = 0
new_image = np.copy(image)
new_seg = np.copy(seg)
new_softmax = np.copy(softmax)
for i in selected_functions:
    trasformation = list_of_transformations[i]
    [new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=trasformation(1,new_image,new_seg,new_softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
    order_of_function = n
    row_single_transformation = [img_name, n, i, [factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value]]
    trasformation_single_image.append(row_single_transformation)
    n += 1

#%%
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(image/255)
# plt.subplot(1,2,2)
# plt.imshow(new_image/255)
# plt.show()


#%%
ric_image = np.copy(new_image)
ric_seg = np.copy(new_seg)
ric_softmax = np.copy(new_softmax)

m = int(n-1)
for i in selected_functions[::-1]:
    rev_transformation = list_of_transformations[i]
    factor = trasformation_single_image[m][3][0]
    factor_sat = trasformation_single_image[m][3][1]
    sign_factor_sat = trasformation_single_image[m][3][2]
    factor_value = trasformation_single_image[m][3][3]
    sign_factor_value =trasformation_single_image[m][3][4]
    [ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = rev_transformation(0,ric_image,ric_seg,ric_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)
    
    m = m-1
    
#%%
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(image/255)
plt.subplot(1,3,2)
plt.imshow(new_image/255)
plt.subplot(1,3,3)
plt.imshow(ric_image/255)
plt.show()

print((image-ric_image).sum())

print(selected_functions)