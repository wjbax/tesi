# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:23:28 2022

@author: ...
"""

# libraries
import os
import mmcv
#import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
from distinctipy import distinctipy #colorblind, examples
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config
import cv2
from tqdm import tqdm
#from util_smooth_fn import predict_img_with_smooth_windowing
#from mmseg.apis import inference_segmentor
from utils_trad import get_concat_h
import cv2
from mmseg.apis import inference_segmentor
import transformations_v2 as t
import pandas as pd
import random
from datetime import datetime

import transformations_v2 as t


###############################################################################
# Perturbation initialization 
###############################################################################

list_of_transformations = {
    "Rotation" :                t.rotation,
    "Vertical mirroring" :      t.vert_mirr,
    "Horizontal mirroring" :    t.hor_mirr,
    "Scaling" :                 t.scaling,
    "Gaussian blurring" :       t.gaussblur,
    "HSV perturbations" :       t.hsv_pert
}

###############################################################################
# Model definition 
###############################################################################

def colors(N):
    BLACK = (0, 0, 0)
    WHITE = (1, 1, 1)
    MAGENTA = (1, 0, 1)
    PURPLE=(0.43742190553354166, 0.169663220790645, 0.5661026334645396)
    ROSE= (0.9100104602105649, 0.5200185698265695, 0.7188660869024343)
    input_colors = [BLACK, WHITE, MAGENTA, PURPLE, ROSE]
    output_colors = distinctipy.get_colors(N,input_colors)
    return  output_colors


classes = ('background','steatosis')
flag_split=0
config_file='configs/knet/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k.py'

# convert dataset annotation to semantic segmentation 
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
#root_dir = '/media/dp-3/HD_aequip/Cytology project (challenge)/OUR DIVISION (1500)'
data_root = os.path.join(root_dir,'DATASET')
img_dir = 'image'
ann_dir = 'anno'

# define class and plaette for better visualization

n_classes=len(classes)
palette=((np.array(colors(n_classes))*255).astype('uint8')).tolist()

if flag_split==1: # 0: use the split defined in the dataset preparation phase, 1: random split 

     split_dir = 'splits'
     mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
     filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
     osp.join(data_root, ann_dir), suffix='.png')]
     train_length = int(len(filename_list)*4/5)

else:
     split_dir = 'splits'
     mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
     filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
     osp.join(data_root, 'train', 'image'), suffix='.png')]
     train_length = int(len(filename_list))
    
###############################################################################
# Dataset definition 
###############################################################################

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config


    
@DATASETS.register_module()
class StanfordBackgroundDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None



# Config files (cfg.keys() - cfg.model)
cfg = Config.fromfile(config_file)
cfg.device='cuda'
#cfg.model.backbone.norm_cfg=dict(type='BN', requires_grad=True)
cfg.model.auxiliary_head.norm_cfg=dict(type='BN', requires_grad=True)
cfg.model.decode_head.kernel_generate_head.norm_cfg=dict(type='BN', requires_grad=True)
cfg.model.decode_head.kernel_update_head[0].num_classes=n_classes
cfg.model.decode_head.kernel_update_head[1].num_classes=n_classes
cfg.model.decode_head.kernel_update_head[2].num_classes=n_classes
cfg.model.decode_head.kernel_generate_head.num_classes=n_classes
cfg.model.auxiliary_head.num_classes=n_classes
cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu = 4
#cfg.optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0005)
# ....
cfg.load_from = 'checkpoints/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k_20220303_154559-d8da9a90.pth'

# Train parameters
iter_for_epoch = round(train_length/cfg.data.samples_per_gpu)
refresh_rate = 4 # define refresh rate of the metrics for each metrics 
num_epochs = 150
refresh_val = 1 # number of epochs to refresh validation metrics
checkpoint_save = 1 # number of epoch for checkpoint saving


# Modify dataset type and path
cfg.dataset_type = 'StanfordBackgroundDataset' # name of the dataset 
cfg.data_root = data_root


cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(416,416)),
    dict(type='PhotoMetricDistortion', brightness_delta=3, contrast_range=(0.90, 1.10), saturation_range=(0.90, 1.10), hue_delta=3),
    # dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **cfg.img_norm_cfg),
    #dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416,416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'
    
cfg.model.auxiliary_head.sampler=dict(type='OHEMPixelSampler', thresh=0.8, min_kept=int((416*416)/4))
cfg.model.decode_head.kernel_generate_head.sampler=dict(type='OHEMPixelSampler', thresh=0.8, min_kept=int((416*416)/4))
  
#weights_cls = getWeights(3, os.path.join(data_root,'train','manual_py'))

cfg.model.decode_head.kernel_generate_head.loss_decode=[dict(type='DiceLoss',loss_name = 'loss_DiceLoss', loss_weight=1.0)]
cfg.model.auxiliary_head.loss_decode=[dict(type='DiceLoss', loss_name = 'loss_DiceLoss',loss_weight=0.4)]

# Set up working dir to save files and logs.
cfg.work_dir = './weights_pretrained_v2/' 

cfg.runner.max_iters = num_epochs*iter_for_epoch
cfg.log_config.interval = round((1/refresh_rate)*iter_for_epoch)
cfg.evaluation.interval = round(refresh_val*iter_for_epoch )
cfg.checkpoint_config.interval = checkpoint_save*iter_for_epoch 

# Set seed to facitate reproducing the result
cfg.seed = 0

# set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


# load my model
# Model inizialization  (ho creato tale funzione modificando il codice originale)  
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
def my_init_segmentor(config, palette, checkpoint=None, device='cuda:0'): 
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = palette
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model 

# load pre-trained model
checkpoint_file = 'weights_pretrained_v2/iter_2280.pth' 
model = my_init_segmentor(cfg,palette, checkpoint=checkpoint_file)





#
#
#
#
#
#
# MODEL INFERENCE
#
#
#
#
#





# Parameters
phase_vector = ['test','val','train']
fatt_resize = 1 # resize for 'debug' figure
input_model = (416,416) # must be a square
n_classes = 2

n_perturbation = 20

# Path
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_root = os.path.join(root_dir,'DATASET')
fold_result_name = 'RESULTS_perturbation'



for phase in phase_vector:
    
    print('\n' + phase.capitalize() + '...')
    mmcv.mkdir_or_exist(fold_result_name)
    mmcv.mkdir_or_exist(os.path.join(fold_result_name,phase,'mask'))
    mmcv.mkdir_or_exist(os.path.join(fold_result_name,phase,'softmax'))

    list_test=os.listdir(os.path.join(data_root,phase,'image'))
    
    for test_image in tqdm(list_test):

        # -------------------------------------------------------------
        # 1. Read image
        # -------------------------------------------------------------
        # tile = mmcv.imread(os.path.join(data_root,phase,'image',test_image))
        
        image = cv2.imread(os.path.join(data_root,phase,'image',test_image))
        image = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        
        b = np.load(os.path.join(data_root,phase,'softmax',test_image))['softmax']
        softmax_topert = np.float32(b[:,:,:])
        original_shape = np.shape(image)
        
        # -------------------------------------------------------------
        # 2. Predict
        # -------------------------------------------------------------
        
        mmcv.mkdir_or_exist(os.path.join(fold_result_name,phase,'softmax',test_image))
        mmcv.mkdir_or_exist(os.path.join(fold_result_name,phase,'mask',test_image))

        for rep in range(0,n_perturbation):
            track_of_transformations = pd.DataFrame(columns=['Dataset','Image_n', 'Order','Transformation','Factor','Factor_sat','Sign_factor_sat','Factor_value','Sign_factor_value'])
            transformation_single_image = []
            selected_functions = random.sample(list(list_of_transformations.keys()), random.randint(1,4))
            if "Scaling" in selected_functions:
                selected_functions.remove("Scaling")
                selected_functions.append("Scaling")
            
            row_track_image = []
            
            n = 0
            new_image = np.copy(image)
            new_softmax = np.copy(softmax_topert)
            for i in selected_functions:
                transformation = list_of_transformations[i]
                [new_image,new_softmax,factor,new_shape,factor_sat,sign_factor_sat,factor_value,sign_factor_value]=transformation(1,new_image,new_softmax,original_shape,new_shape=[],factor=0,factor_sat=0,sign_factor_sat=1,factor_value=0,sign_factor_value=1)
                order_of_function = n
                row_single_transformation = [test_image, n, i, [factor,factor_sat,sign_factor_sat,factor_value,sign_factor_value]]
                transformation_single_image.append(row_single_transformation)
                n += 1
                row_data_dict = {'Dataset' : data_root, 'Image_n' : rep, 'Order' : n, 'Transformation' : i, 'Factor' : factor, 'Factor_sat' : factor_sat, 'Sign_factor_sat' : sign_factor_sat, 'Factor_value' : factor_value, 'Sign_factor_value' : sign_factor_value}
                row_track_image = pd.DataFrame([row_data_dict])
                track_of_transformations=pd.concat([track_of_transformations,row_track_image],axis=0)
        
            
            # image perturbation (tile_perturb)
            tile_perturb = np.copy(new_image)
            
            # make prediction on perturbated image 
            result = inference_segmentor(model, tile_perturb)
            softmax = np.transpose(np.array(result[0]),axes=[1,2,0])
            
            # restore softmax based on the perturbation applied
            ric_image = np.copy(new_image)
            ric_softmax = np.copy(softmax)
    
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
            
            # Save softmax
            np.savez_compressed(os.path.join(fold_result_name,phase,'softmax',test_image,str(rep+1)+'.npz'), softmax = ric_softmax)
   
