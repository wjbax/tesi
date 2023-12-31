# transformations -> I suppose width and height of the 3 images are the same,
# I will put an if clause to disclose the image if this is not the case

import numpy as np
from skimage import transform
import cv2
from PIL import Image, ImageFilter
import random

#%% Rotation
def rotation(direction,image,softmax,original_shape=[],new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1):
    if direction == 1:
        factor = random.sample([90,270],1)[0]
        new_image = transform.rotate(image,factor)
        new_softmax = transform.rotate(softmax,factor)

    elif direction == 0:
        new_image = transform.rotate(image,-factor)
        new_softmax = transform.rotate(softmax,-factor)

    else:
        raise Exception("Error! Direction must be 1 (direct) or 0 (inverse)")
    new_image = np.round(new_image).astype(np.float32)
    # new_softmax = np.round(new_softmax).astype(np.float32)
    return new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value


#%% Scaling
def scaling(direction,image,softmax,original_shape=[],new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1):
    width = original_shape[0]
    height = original_shape[1]
    if direction == 1:
        factor = np.random.uniform(0.8,1)
        new_shape = [np.ceil(factor*np.array(width)).astype(int),np.ceil(factor*np.array(height)).astype(int)]
        x = width - new_shape[0]
        y = height - new_shape[1]
        
        temp_image = transform.resize(image, np.append(new_shape,[3]))
        temp_softmax = transform.resize(softmax, np.append(new_shape,[2]))
        
        new_image = np.zeros([width,height,3])
        new_softmax = np.zeros([width,height,2])
        
        new_image[x:,y:,:] = temp_image
        new_softmax[x:,y:,:] = temp_softmax
        
    elif direction == 0:
        x = width - new_shape[0]
        y = height - new_shape[1]
        crop_image = image[x:,y:,:]
        new_image = transform.resize(crop_image, [width,height,3])
        crop_softmax = softmax[x:,y:,:]
        new_softmax = transform.resize(crop_softmax, [width,height,2])
    else:
        raise Exception("Error! Direction must be 1 (direct) or 0 (inverse)")
    
    new_image = np.round(new_image).astype(np.float32)
    # new_softmax = np.round(new_softmax).astype(np.float32)
    return new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value

#%% Gaussian Blur
def gaussblur(direction,image,softmax,original_shape=[],new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1):
    if direction == 1:
        radius = 2
        image = Image.fromarray(image.astype(np.uint8))
        noise = ImageFilter.GaussianBlur(radius)
        new_image = image.filter(noise)
        new_image = np.float32(np.array(new_image))
        new_softmax = softmax
    elif direction == 0:
        new_image = image
        new_softmax = softmax
    else:
        raise Exception("Error! Direction must be 1 (direct) or 0 (inverse)")
    return new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value

#%% Vertical_mirroring
def vert_mirr(direction,image,softmax,original_shape=[],new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1):
    if direction == 1:
        new_image = image[::-1,:,:]
        new_softmax = softmax[::-1,:,:]
    elif direction == 0:
        new_image = image[::-1,:,:]
        new_softmax = softmax[::-1,:,:]
    else:
        raise Exception("Error! Direction must be 1 (direct) or 0 (inverse)")
    new_image = np.round(new_image).astype(np.float32)
    # new_softmax = np.round(new_softmax).astype(np.float32)
    return new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value

#%% Horizontal_mirroring
def hor_mirr(direction,image,softmax,original_shape=[],new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1):
    if direction == 1:
        new_image = image[:,::-1,:]
        new_softmax = softmax[:,::-1,:]
    elif direction == 0:
        new_image = image[:,::-1,:]
        new_softmax = softmax[:,::-1,:]
    else:
        raise Exception("Error! Direction must be 1 (direct) or 0 (inverse)")
    
    new_image = np.round(new_image).astype(np.float32)
    # new_softmax = np.round(new_softmax).astype(np.float32)
    return new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value

#%% HSV_perturbations
def hsv_pert(direction,image,softmax,original_shape=[],new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1):
    if direction == 1:
        factor = np.random.randint(-10,10)
        new_image_raw = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        new_image = np.copy(new_image_raw)
    
        new_hue = (new_image_raw[:,:,0] + factor) % 360
        new_sat = new_image_raw[:,:,1]
        new_value = new_image_raw[:,:,2]
    
        max_sat = new_sat.max()
        min_sat = new_sat.min()
        diffmax_sat = np.float32(1 - max_sat)
        if diffmax_sat > min_sat:
            max_perc_sat = 0.15*(max_sat-min_sat)
            sign_factor_sat = 1
            factor_sat = np.min([np.random.uniform(0,diffmax_sat),max_perc_sat])
            new_sat = new_sat + factor_sat
        else:
            sign_factor_sat = 0
            factor_sat = np.random.uniform(0,min_sat)
            new_sat = new_sat - factor_sat
    
        max_value = new_value.max()
        min_value = new_value.min()
        diffmax_value = 255 - max_value
        if diffmax_value > min_value:
            max_perc_value = 0.15*(max_value-min_value)
            sign_factor_value = 1
            factor_value = np.min([np.random.randint(0,diffmax_value),max_perc_value])
            new_value = new_value + factor_value
        else:
            sign_factor_value = 0
            factor_value = np.random.randint(0,min_value)
            new_value = new_value - factor_value
            
        new_image[:,:,0] = new_hue
        new_image[:,:,1] = new_sat
        new_image[:,:,2] = new_value

        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        
        new_softmax = softmax
        
    elif direction == 0:
        new_image_raw = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        new_image_ric = np.copy(new_image_raw)
    
        ric_hue = (new_image_ric[:,:,0] + 360 - factor) % 360
    
        ric_sat = np.copy(new_image_ric[:,:,1])
        if sign_factor_sat == 1:
            ric_sat = ric_sat - factor_sat
        else:
            ric_sat = ric_sat + factor_sat
            
        ric_value = np.copy(new_image_ric[:,:,2])
        if sign_factor_value == 1:
            ric_value = ric_value - factor_value
        else:
            ric_value = ric_value + factor_value
    
        new_image_ric[:,:,0] = ric_hue
        new_image_ric[:,:,1] = ric_sat
        new_image_ric[:,:,2] = ric_value
    
        new_image_ric = cv2.cvtColor(new_image_ric, cv2.COLOR_HSV2RGB)
        new_image_ric = np.round(new_image_ric).astype(np.float32)
        
        new_image = new_image_ric
        
        new_softmax = softmax
        
    else:
        raise Exception("Error! Direction must be 1 (direct) or 0 (inverse)")
    return new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value

#%%
