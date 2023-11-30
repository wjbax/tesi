import os
import numpy as np
import cv2
import transformations
import pandas as pd
import random
import transformations as t
import matplotlib.pyplot as plt
from PIL import Image

#%% Dataset path and directories
DATASET_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/"
directories = [d for d in os.listdir(DATASET_path) if os.path.isdir(os.path.join(DATASET_path, d))]
## IMPORTANT: I changed the structure of the dataset splitting the Prostate HE (glands, tumor)
## directory into Prostate HE glands and Prostate HE tumor directories, also changing the name
## of the DATASET directory present in them from DATASET_glands or DATASET_tumor to DATASET to
## match the structure of the other directories: we have 5 directories now

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
for directory in directories:
    
    transformations_track_table = pd.DataFrame(columns=['IMG_Name', 'N_of_transformation', 'Transformation', 'Parameter(s)'])
    
    img_directory = DATASET_path + directory + '/DATASET/train/train/image/'
    img_saving_directory = DATASET_path + directory + '/DATASET/train/train/perturbated_images/'
    if not os.path.exists(img_saving_directory):
        os.makedirs(img_saving_directory)
    img_list = os.listdir(img_directory)
    
    seg_directory = DATASET_path + directory + "/DATASET/train/train/manual/"
    seg_saving_directory =  DATASET_path + directory + '/DATASET/train/train/perturbated_segmentations/'
    if not os.path.exists(seg_saving_directory):
        os.makedirs(seg_saving_directory)
        # seg_list = os.listdir(seg_directory)
    
    softmax_directory = DATASET_path + directory + "/k-net+swin/TEST_2classes/RESULTS/train/softmax/"
    softmax_saving_directory = DATASET_path + directory + '/DATASET/train/train/perturbated_softmax/'
    if not os.path.exists(softmax_saving_directory):
        os.makedirs(softmax_saving_directory)
        # softmax_list = os.listdir(softmax_directory)
        
    
    for img_name in img_list:
        
        track_of_transformations = pd.DataFrame(columns=['Image_n', 'Order','Transformation','Parameters'])
        
        pert_image_directory = img_saving_directory + img_name[:-4] + "/"
        if not os.path.exists(pert_image_directory):
            os.makedirs(pert_image_directory)
        pert_seg_directory = seg_saving_directory + img_name[:-4] + "/"
        if not os.path.exists(pert_seg_directory):
            os.makedirs(pert_seg_directory)
        pert_softmax_directory = softmax_saving_directory + img_name[:-4] + "/"
        if not os.path.exists(pert_softmax_directory):
            os.makedirs(pert_softmax_directory)

        
        image_path = img_directory + img_name
        seg_path = seg_directory + img_name
        softmax_name = img_name[:-3] + "npz"
        softmax_path = softmax_directory + softmax_name
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        
        seg = cv2.imread(seg_path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        seg = np.float32(seg)
        
        b = np.load(softmax_path)['softmax']
        softmax = np.float32(b[:,:,1])
        original_shape = np.shape(image)
        #da qui ciclo for di 20 volte
        for counter in range(20):
            trasformation_single_image = []
            selected_functions = random.sample(list(list_of_transformations.keys()), random.randint(1,4))
            if "Scaling" in selected_functions:
                selected_functions.remove("Scaling")
                selected_functions.append("Scaling")
            
            row_track_image = []
            
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
                row_data_dict = {'Image_n' : counter, 'Order' : n, 'Transformation' : i, 'Parameters' : [factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value]}
                row_track_image = pd.DataFrame(row_data_dict)
                track_of_transformations=pd.concat([track_of_transformations,row_track_image],axis=0)
            
            # salvataggio delle immagini
            file_name = img_name[:-4] + "_" + str(counter) + ".tiff"
            
            new_image_tosave = cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(pert_image_directory + file_name, new_image_tosave)
            cv2.imwrite(pert_seg_directory + file_name, new_seg)
            cv2.imwrite(pert_softmax_directory + file_name, new_softmax)
            
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
            print(selected_functions)
            print(selected_functions[::-1])
            diff=(image-ric_image)/255
            print(abs(diff.sum()))
            plt.figure(figsize=(10,5))
            plt.subplot(1,4,1)
            plt.imshow(image/255)
            plt.subplot(1,4,2)
            plt.imshow(new_image/255)
            plt.subplot(1,4,3)
            plt.imshow(ric_image/255)
            plt.subplot(1,4,4)
            plt.imshow(diff)
            plt.show()
            
        #qui finisce il ciclo for di 20 volte
        if 4>2 : break
    if 4>2 : break
#%%
# print(selected_functions)
# print(selected_functions[::-1])
# diff=(image-ric_image)/255
# print(diff.sum())
# plt.figure(figsize=(10,5))
# plt.subplot(1,4,1)
# plt.imshow(image/255)
# plt.subplot(1,4,2)
# plt.imshow(new_image/255)
# plt.subplot(1,4,3)
# plt.imshow(ric_image/255)
# plt.subplot(1,4,4)
# plt.imshow(diff)
# plt.show()