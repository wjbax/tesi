import os
import numpy as np
import cv2
import transformations
import pandas as pd
import random
from datetime import datetime

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
current_time = datetime.now()
hour = current_time.hour
minute = current_time.minute
second = current_time.second
print(f"Start time: {hour}:{minute}:{second}")

total_transformations_track_table = pd.DataFrame(columns=['Dataset','Image_n', 'Order','Transformation','Factor','Factor_sat','Sign_factor_sat','Factor_value','Sign_factor_value'])
for directory in directories:
    # if directory == "Liver steatosis HE": continue
    if directory == "Prostate HE glands": continue
    if directory == "Prostate HE tumor": continue
    if directory == "Renal PAS glomeruli (seg, cls)": continue
    if directory == "Renal PAS tubuli (seg, cls)": continue

    current_time = datetime.now()
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    print(f"Start time for image {directory}: {hour}:{minute}:{second}")
    
    directory_transformations_track_table = pd.DataFrame(columns=['Dataset','Image_n', 'Order','Transformation','Factor','Factor_sat','Sign_factor_sat','Factor_value','Sign_factor_value'])
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
        
        current_time = datetime.now()
        hour = current_time.hour
        minute = current_time.minute
        second = current_time.second
        print(f"Start time for image {img_name}: {hour}:{minute}:{second}")
        
        track_of_transformations_total = pd.DataFrame(columns=['Dataset','Image_n', 'Order','Transformation','Factor','Factor_sat','Sign_factor_sat','Factor_value','Sign_factor_value'])
        
        pert_image_directory = img_saving_directory + img_name[:-4] + "/"
        if not os.path.exists(pert_image_directory):
            os.makedirs(pert_image_directory)
        pert_seg_directory = seg_saving_directory + img_name[:-4] + "/"
        if not os.path.exists(pert_seg_directory):
            os.makedirs(pert_seg_directory)
        pert_softmax_directory = softmax_saving_directory + img_name[:-4] + "/"
        if not os.path.exists(pert_softmax_directory):
            os.makedirs(pert_softmax_directory)
            
        ric_image_directory = img_saving_directory + "ric_" +img_name[:-4] + "/"
        if not os.path.exists(ric_image_directory):
            os.makedirs(ric_image_directory)
        ric_seg_directory = seg_saving_directory + "ric_" + img_name[:-4] + "/"
        if not os.path.exists(ric_seg_directory):
            os.makedirs(ric_seg_directory)
        ric_softmax_directory = softmax_saving_directory + "ric_" + img_name[:-4] + "/"
        if not os.path.exists(ric_softmax_directory):
            os.makedirs(ric_softmax_directory)

        
        image_path = img_directory + img_name
        seg_path = seg_directory + img_name
        softmax_name = img_name[:-4] + ".npz"
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
            track_of_transformations = pd.DataFrame(columns=['Dataset','Image_n', 'Order','Transformation','Factor','Factor_sat','Sign_factor_sat','Factor_value','Sign_factor_value'])
            transformation_single_image = []
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
                transformation = list_of_transformations[i]
                [new_image,new_seg,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=transformation(1,new_image,new_seg,new_softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
                order_of_function = n
                row_single_transformation = [img_name, n, i, [factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value]]
                transformation_single_image.append(row_single_transformation)
                n += 1
                row_data_dict = {'Dataset' : directory, 'Image_n' : counter, 'Order' : n, 'Transformation' : i, 'Factor' : factor, 'Factor_sat' : factor_sat, 'Sign_factor_sat' : sign_factor_sat, 'Factor_value' : factor_value, 'Sign_factor_value' : sign_factor_value}
                row_track_image = pd.DataFrame([row_data_dict])
                track_of_transformations=pd.concat([track_of_transformations,row_track_image],axis=0)
            
            # salvataggio delle immagini
            file_name = img_name[:-4] + "_" + str(counter) + ".png"
            
            new_image_tosave = cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(pert_image_directory + file_name, new_image_tosave)
            cv2.imwrite(pert_seg_directory + file_name, new_seg)
            np.savez(pert_softmax_directory + file_name[:-4] + ".npz",softmax=new_softmax)
            
            ric_image = np.copy(new_image)
            ric_seg = np.copy(new_seg)
            ric_softmax = np.copy(new_softmax)
    
            m = int(n-1)
            for i in selected_functions[::-1]:
                rev_transformation = list_of_transformations[i]
                factor = transformation_single_image[m][3][0]
                factor_sat = transformation_single_image[m][3][1]
                sign_factor_sat = transformation_single_image[m][3][2]
                factor_value = transformation_single_image[m][3][3]
                sign_factor_value =transformation_single_image[m][3][4]
                [ric_image,ric_seg,ric_softmax,_,_,_,_,_,_] = rev_transformation(0,ric_image,ric_seg,ric_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)
                m = m-1
                
            # salvataggio delle immagini
            ric_file_name = "ric_" + img_name[:-4] + "_" + str(counter) + ".png"
            
            ric_image_tosave = cv2.cvtColor(ric_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(ric_image_directory + ric_file_name, ric_image_tosave)
            cv2.imwrite(ric_seg_directory + ric_file_name, ric_seg)
            np.savez(ric_softmax_directory + ric_file_name[:-5],softmax=ric_softmax)
            
            track_of_transformations.to_csv(pert_image_directory+'transformations_for_image_'+file_name[:-4]+'.csv', index=False)
            track_of_transformations.to_csv(pert_seg_directory+'transformations_for_seg_'+file_name[:-4]+'.csv', index=False)
            track_of_transformations.to_csv(pert_softmax_directory+'transformations_for_softmax_'+file_name[:-4]+'.csv', index=False)
        #qui finisce il ciclo for di 20 volte
        
        track_of_transformations_total=pd.concat([track_of_transformations_total,track_of_transformations],axis=0)
        # if 4>2 : break
    track_of_transformations_total.to_csv(img_saving_directory+'transformations_for_dataset_'+directory+'.csv', index=False)
    current_time = datetime.now()
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    print(f"Stop time for image {directory}: {hour}:{minute}:{second}")
    # if 4>2 : break
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