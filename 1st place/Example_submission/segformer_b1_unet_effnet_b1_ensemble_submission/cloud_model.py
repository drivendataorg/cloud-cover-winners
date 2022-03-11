
from pathlib import Path
import random
from pprint import pprint

import torch
import torch.nn as nn
from timm.models.efficientnet import *
import segmentation_models_pytorch as smp
import rasterio
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import time
import os
import torchvision

# These transformations will be passed to our model class
class CloudDataset(torch.utils.data.Dataset):

    def __init__(self, chip_ids_df, 
                 x_path = '../data/train_features/', 
                 y_path= '../data/train_labels/', 
                 bands=[4,3,2],transforms=None):
        self.data = chip_ids_df
        self.data_path = x_path
        self.label_path = y_path
        self.bands = bands
        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        img = self.data.iloc[idx]
        chip_id = img.chip_id
        imgs = []
        for b in self.bands:
            pth = f'{self.data_path}/{chip_id}/B0{b}.tif'
            with rasterio.open(pth) as img_file:
                img = img_file.read(1).astype(float)
                img = (img/2**16).astype(np.float32)
                imgs.append(img)
        x_arr= np.stack(imgs,axis=-1)
        
        x_arr = np.transpose(x_arr, [2, 0, 1])
        sample = {"chip_id": chip_id, "chip": x_arr}

        return sample
    
class Net4CH(nn.Module):
    def __init__(self,params):
        super(Net4CH, self).__init__()

        aux_params=dict(
                        pooling='avg',             # one of 'avg', 'max'
                        dropout=0.3,               # dropout ratio, default is None
                        activation=None,      # activation function, default is None
                        classes=1,
                    ) 
        self.unet = smp.Unet(
                    encoder_name=params['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=params['weights'],     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    decoder_attention_type= None,                      # model output channels (number of classes in your dataset)
                    classes=1,aux_params=aux_params
                    )



    # @torch.cuda.amp.autocast()
    def forward(self, image):
        batch_size = len(image)
        mask,logit = self.unet(image)
        return mask
