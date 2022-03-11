
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
        self.max_values = {4:23104,3:26096,2:27600,8:19568}

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
    


 


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead

class UnetEffNetV2(nn.Module):
    def __init__(self,params):
        super(UnetEffNetV2, self).__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        self.backbone = timm.create_model('tf_efficientnetv2_b2', features_only=True, 
                                          out_indices=[0,1,2,3],pretrained=True)
        self.decode_head = UnetDecoder(
                            encoder_channels=[16, 32, 56, 120],
                            decoder_channels=[16, 32, 56, 120],
                            n_blocks=4,
                            use_batchnorm=True,
                            center=False,
                            attention_type=None
                        )
        
        self.segment_classifier = SegmentationHead(56,1,upsampling=4)

        
    def forward(self,image):
        image = image[:,0:3]
        x = self.backbone(image)
        x=self.decode_head(*x)
        x=self.segment_classifier(x)
        x = F.interpolate(x, image.shape[-2:], mode="bilinear", align_corners=True)
        return x
    

class UNetEFF1_4CH(nn.Module):
    def __init__(self,params):
        super(UNetEFF1_4CH, self).__init__()

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
