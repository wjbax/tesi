# this is the main script

# please make sure to adjust:
#     DATASET_path (row 22)
#     Code for inference (row 141)
 
# Please note that:
# This code works based on the fact that the image
# and the softmax have the same size for the first
# two dimensions. It does not check this condition,
# because I encountered no couple of image-softmax
# not satisfatying that particular condition

#%% Import of libraries used

import os
import numpy as np
import cv2
import transformations_v2 as t
import pandas as pd
import random
from datetime import datetime

#%% Dataset path and directories
DATASET_path = "D:/DATASET TESI/Bassolino (XAI-UQ segmentation)/Bassolino (XAI-UQ segmentation)/"
directories = [d for d in os.listdir(DATASET_path) if os.path.isdir(os.path.join(DATASET_path, d))]

#%% List of transformations that could be applied
#  please note that any transformation has the same chance
#  to be chosen, and when chosen there is 100% chance
#  of being applied

#  In future versions, I could easily change this behaviour,
#  but for sake of testing I think going simple in this is
#  the best choice

#  please, refer to the module transformations.py for
#  better understanding on how these transformations work

list_of_transformations = {
    "Rotation" :                t.rotation,
    "Vertical mirroring" :      t.vert_mirr,
    "Horizontal mirroring" :    t.hor_mirr,
    "Scaling" :                 t.scaling,
    "Gaussian blurring" :       t.gaussblur,
    "HSV perturbations" :       t.hsv_pert
}

#%% body

#%%% initialization of time stamps and total track table 
# (the latter will be useful when working on more datasets)

current_time = datetime.now()
hour = current_time.hour
minute = current_time.minute
second = current_time.second
print(f"Start time: {hour}:{minute}:{second}")

total_transformations_track_table = pd.DataFrame(columns=['Dataset','Image_n', 'Order','Transformation','Factor','Factor_sat','Sign_factor_sat','Factor_value','Sign_factor_value'])

#%%% for cycle for the various datasets

for directory in directories:
    # the next 5 lines ensure this code only applies to the 
    # Liver Steatosis HE Dataset; when removing/commenting
    # the ifs, other dataset will be worked on.
    # please do not remove/comment, because the way the other
    # dataset are built makes the code NOT work anymore 
    # (file names are built differently)
    
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
    
    img_directory={"train" : [],"val" : [],"test": []}
    img_list={"train" : [],"val" : [],"test": []}
    
    softmax_directory={"train" : [],"val" : [],"test": []}
    softmax_saving_directory={"train" : [],"val" : [],"test": []}
    
    softmax_results_directory = "/k-net+swin/TEST_2classes/RESULTS_PERTURBATIONS/"
    if not os.path.exists(softmax_results_directory):
        os.makedirs(softmax_results_directory)
    
    internal_set = ["train","val","test"]
    for i_set in internal_set:
        img_directory[i_set] = DATASET_path + directory + '/DATASET/' + i_set + '/' + i_set + '/image/'
        img_list[i_set] = os.listdir(img_directory[i_set])
        

        softmax_directory[i_set] = DATASET_path + directory + "/k-net+swin/TEST_2classes/RESULTS/" + i_set + "/softmax/"
        softmax_saving_directory[i_set] = DATASET_path + directory + softmax_results_directory + i_set + "/softmax/"
        if not os.path.exists(softmax_saving_directory[i_set]):
            os.makedirs(softmax_saving_directory[i_set])
        
    #%%% for cycle to work on every image of the train set
    for i_set in internal_set:     
        for img_name in img_list[i_set]:
            
            current_time = datetime.now()
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second
            print(f"Start time for image {img_name}, from the {i_set} set: {hour}:{minute}:{second}")
            
            track_of_transformations_total = pd.DataFrame(columns=['Dataset','Image_n', 'Order','Transformation','Factor','Factor_sat','Sign_factor_sat','Factor_value','Sign_factor_value'])
    
            pert_softmax_directory = softmax_saving_directory[i_set] + img_name[:-4] + "/"
            if not os.path.exists(pert_softmax_directory):
                os.makedirs(pert_softmax_directory)
            
            image_path = img_directory[i_set] + img_name
            softmax_name = img_name[:-4] + ".npz"
            softmax_path = softmax_directory[i_set] + softmax_name
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            
            b = np.load(softmax_path)['softmax']
            softmax = np.float32(b[:,:,:])
            original_shape = np.shape(image)
            
            # for cycle to perturb 20 times the same image
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
                new_softmax = np.copy(softmax)
                for i in selected_functions:
                    transformation = list_of_transformations[i]
                    [new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=transformation(1,new_image,new_softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
                    order_of_function = n
                    row_single_transformation = [img_name, n, i, [factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value]]
                    transformation_single_image.append(row_single_transformation)
                    n += 1
                    row_data_dict = {'Dataset' : directory, 'Image_n' : counter, 'Order' : n, 'Transformation' : i, 'Factor' : factor, 'Factor_sat' : factor_sat, 'Sign_factor_sat' : sign_factor_sat, 'Factor_value' : factor_value, 'Sign_factor_value' : sign_factor_value}
                    row_track_image = pd.DataFrame([row_data_dict])
                    track_of_transformations=pd.concat([track_of_transformations,row_track_image],axis=0)
            
                
                
                
                
                
                #  INSERT HERE THE CODE FOR THE INFERENCE OF ONE OF THE 20 IMAGES:
                #  WE ARE INSIDE THE FOR CYCLE, SO ANY INFERENCE DONE HERE WILL BE 
                #  DONE ON ONE OF THE 20 IMAGES THAT COME FROM THE PERTURBATION OF
                #  A SINGLE IMAGE.
                
                
                
                
                
                
                
                ric_image = np.copy(new_image)
                ric_softmax = np.copy(new_softmax)
        
                m = int(n-1)
                for i in selected_functions[::-1]:
                    rev_transformation = list_of_transformations[i]
                    factor = transformation_single_image[m][3][0]
                    factor_sat = transformation_single_image[m][3][1]
                    sign_factor_sat = transformation_single_image[m][3][2]
                    factor_value = transformation_single_image[m][3][3]
                    sign_factor_value =transformation_single_image[m][3][4]
                    [ric_image,ric_softmax,_,_,_,_,_,_] = rev_transformation(0,ric_image,ric_softmax,original_shape,new_shape,factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value)
                    m = m-1
                    
                # saving the "ricreated" softmax
                file_name = img_name[:-4] + "_" + str(counter) + ".png"
                np.savez(pert_softmax_directory + file_name[:-4] + ".npz",softmax=ric_softmax)   
                track_of_transformations.to_csv(pert_softmax_directory+'transformations_for_softmax_'+file_name[:-4]+'.csv', index=False)
            # ending of for cycle for one single image
            track_of_transformations_total=pd.concat([track_of_transformations_total,track_of_transformations],axis=0)
    
        track_of_transformations_total.to_csv(softmax_saving_directory[i_set]+'transformations_for_dataset_'+directory+'.csv', index=False)
        current_time = datetime.now()
        hour = current_time.hour
        minute = current_time.minute
        second = current_time.second
        print(f"Stop time for {directory}, {i_set} set: {hour}:{minute}:{second}")