#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path
import random
from pprint import pprint

import cv2
import rasterio
import segmentation_models_pytorch as smp
import albumentations
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
# These transformations will be passed to our model class
import torch
import yaml
from tqdm.auto import tqdm


# In[4]:


def intersection_over_union_np(pred, true):

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    
    print(intersection.sum() , union.sum())
    return intersection.sum() / union.sum()


# In[5]:


import rasterio as rasterio
import torch
import torch.nn.functional as F
import numpy as np

class CloudDataset(torch.utils.data.Dataset):

    def __init__(self, chip_ids_df, 
                 x_path = './', 
                 y_path= './', 
                 bands=[4,3,2,8],transforms=None):
        self.data = chip_ids_df
        self.data_path = x_path
        self.label_path = y_path
        self.bands = bands
        self.transforms = transforms
        self.max_values = {4:23104,3:26096,2:27600,8:19568}

    def __len__(self):
        return len(self.data)
    

    def normalize_data_numpy(self, data, pixel_max=1):
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        range_val = max_val - min_val
        data = (data.copy()-min_val)/range_val
        data *= pixel_max
        return data
    
    
    def getMetaLabel(self,lbl):
        lbl_sum = lbl.sum()
        lbl_shp = lbl.shape
        if lbl_sum == lbl_shp[0]*lbl_shp[1]:
            return 1
        else:
            return 0
    
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
        
        if self.label_path is not None: 

            lpth = f'{self.label_path}/{chip_id}.tif'
            with rasterio.open(lpth) as lp:
                y_arr = lp.read(1).astype(int)
            if self.transforms:
                sample_d = self.transforms(image=x_arr,mask=y_arr)
                x_arr=sample_d["image"]
                y_arr=sample_d["mask"]
                
            x_arr = np.transpose(x_arr, [2, 0, 1])
            meta_label = self.getMetaLabel(y_arr)
            sample = {"chip_id": chip_id, "chip": x_arr, "label":y_arr, "Bkg_label":meta_label}
            del img
        else:
            x_arr = np.transpose(x_arr, [2, 0, 1])
            sample = {"chip_id": chip_id, "chip": x_arr}

        return sample


# In[6]:


import torch.nn as nn
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input, target):
        #ce_loss = F.binary_cross_entropy_with_logits(input.squeeze(1), target.long(),reduction=self.reduction,weight=self.weight)
        ce_loss = self.bce(input.squeeze(1), target.float(),)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)#.mean()
        return focal_loss


# In[8]:


class XEDiceLoss(torch.nn.Module):
    """
    Computes (0.5 * CrossEntropyLoss) + (0.5 * DiceLoss).
    """

    def __init__(self):
        super().__init__()
        self.xe = torch.nn.BCEWithLogitsLoss(reduction="none")
        #self.xe = FocalLoss(reduction='none')

    def forward(self, pred, true):
        
        pred = pred.squeeze(1)
        valid_pixel_mask = true.ne(255)  # valid pixel mask

        # Cross-entropy loss
        temp_true = torch.where((true == 255), 0, true)  # cast 255 to 0 temporarily
        xe_loss = self.xe(pred, temp_true.float())
        xe_loss = xe_loss.masked_select(valid_pixel_mask).mean()

        # Dice loss
        
        pred = pred.sigmoid() #torch.softmax(pred, dim=1)[:, 1]
        
        pred = pred.masked_select(valid_pixel_mask)
        true = true.masked_select(valid_pixel_mask)
        
        dice_loss = 1 - (2.0 * torch.sum(pred * true) + 1e-7) / (torch.sum(pred + true) + 1e-7)
        
        #print(xe_loss,dice_loss,(2.0 * torch.sum(pred * true) + 1e-7))

        return (0.5 * xe_loss) + (0.5 * dice_loss)


# In[10]:


def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    pred = pred.squeeze(1)
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask)
    pred = pred.masked_select(valid_pixel_mask)

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum(), union.sum()


# In[11]:

from timm.models.efficientnet import *
import segmentation_models_pytorch as smp

class Net(nn.Module):
    def __init__(self,params):
        super(Net, self).__init__()

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


# In[12]:


import torch.cuda.amp as amp
class AmpNet(Net):
    
    def __init__(self,params):
        super(AmpNet, self).__init__(params)
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)

is_mixed_precision = True  #True #False


# In[13]:


def getModel(params):
    unet_model = AmpNet(params)
    if GPU:
        unet_model.cuda()
    return unet_model

def getOptimzersScheduler(model,params,steps_in_epoch=25,pct_start=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,steps_per_epoch=1,
                                                    pct_start=pct_start,
                                                    max_lr=params['learning_rate'],
                                                    epochs  = params['max_epochs'], 
                                                    div_factor = params['div_factor'], 
                                                    final_div_factor=params['final_div_factor'],
                                                    verbose=True)
    
    return optimizer,scheduler,False
    


# In[14]:


def save_model(epoch,model,ckpt_path='./',name='unet_effnet_b1',val_iou=0):
    path = os.path.join(ckpt_path, '{}_wo_ca.pth'.format(name))
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)
    
def load_model(model,ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state)
    return model


# In[15]:


def getDataLoader(data_path,params,train_x,val_x,train_transforms=None,val_transforms=None):
    
    train_dataset = CloudDataset(
            chip_ids_df=train_x,x_path=f'{data_path}/train_features',y_path=f'{data_path}/train_labels', transforms=train_transforms
        )
    val_dataset = CloudDataset(val_x,x_path=f'{data_path}/train_features',y_path=f'{data_path}/train_labels',  transforms=val_transforms)
    
    trainDataLoader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=params['batch_size'],
                            num_workers=params['num_workers'],
                            shuffle=True,
                            pin_memory=False,
                            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id)
                        )
    valDataLoader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=params['batch_size']*2,
                        num_workers=params['num_workers'],
                        shuffle=False,
                        pin_memory=False,
                    )
    
    return trainDataLoader,valDataLoader


# In[16]:


def training_step(model, batch, batch_idx,optimizer,scheduler,isStepScheduler=False):
    # Load images and labels
    x = batch["chip"].float()
    y = batch["label"].long()
    image_label = batch["Bkg_label"].long()
    
    if GPU:
        x, y, image_label = x.cuda(non_blocking=True), y.cuda(non_blocking=True), image_label.cuda(non_blocking=True)

    criterion = XEDiceLoss()
    criterionBkg = torch.nn.BCEWithLogitsLoss(reduction="mean") 
    
    optimizer.zero_grad()
    # Forward 
    if is_mixed_precision:
        with amp.autocast():
            preds = model(x)
            
            loss = criterion(preds, y)
            #image_loss = criterionBkg(logit.squeeze(dim=-1),image_label.float())
            
            #loss = 0.8*loss + 0.2*image_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
            loss = loss.item()
    else:
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        
    if isStepScheduler:
        scheduler.step()

    # Calculate validation IOU (global)
    preds = (preds.detach().sigmoid().cpu() > 0.5) *1
    y = y.detach().cpu()
    intersection, union = intersection_and_union(preds, y)
    
    
    return loss,intersection.cpu().numpy(), union.cpu().numpy()

def validation_step(model, batch, batch_idx):
    # Load images and labels
    x = batch["chip"].float()
    y = batch["label"].long()
    image_label = batch["Bkg_label"].long()
    if GPU:
        x, y, image_label = x.cuda(non_blocking=True), y.cuda(non_blocking=True), image_label.cuda(non_blocking=True)

    criterion = XEDiceLoss()
    criterionBkg = torch.nn.BCEWithLogitsLoss(reduction="mean") 
    
    # Forward pass & softmax
    with torch.no_grad():
        if is_mixed_precision:
            with amp.autocast():
                preds = model(x)
                xe_dice_loss = criterion(preds, y)
                #image_loss = criterionBkg(logit.squeeze(dim=-1),image_label.float())
                loss = xe_dice_loss #+ image_loss
        else:
            preds = model(x)
            loss = criterion(preds, y)
        
    preds = preds.sigmoid() #torch.softmax(preds, dim=1)[:, 1]
    preds_hard = (preds > 0.5) * 1
    intersection, union = intersection_and_union(preds_hard.cpu(), y.cpu())
    
    loss = loss.item()
    
    return loss,intersection.cpu().numpy(), union.cpu().numpy()


# In[17]:


def train_epoch(model,trainDataLoader,optimizer,scheduler,isStepScheduler=True):
    total_intersection=0
    total_union=0
    total_loss=0
    model.train()
    torch.set_grad_enabled(True)
    total_step=0
    ious = []
    
    
    pbar = tqdm(enumerate(trainDataLoader),total=len(trainDataLoader))
    for bi,data in pbar:
        loss,intersection, union = training_step(model,data,bi,optimizer,scheduler)
        total_intersection+=intersection
        total_union+=union
        
        ious.append(intersection/union)
        total_loss+=loss
        total_step+=1
        iou=total_intersection/total_union
        pbar.set_postfix({'iou': iou,'iou_mean':np.mean(ious),'loss':total_loss/total_step})
        
    if not isStepScheduler: #in case epoch based scheduler
        scheduler.step()
            
    total_loss /= total_step
    iou = total_intersection/total_union #np.mean(ious)#total_intersection/total_union
    return iou,total_loss
        

def val_epoch(model,valDataLoader):
    total_intersection=0
    total_union=0
    total_loss=0
    
    total_step=0
    model.eval()
    ious = []
    pbar=tqdm(enumerate(valDataLoader),total=len(valDataLoader))
    for bi,data in pbar :
        loss,intersection, union = validation_step(model,data,bi)
        total_intersection+=intersection
        total_union+=union
        if intersection >0:
            ious.append(intersection/union)
        total_loss+=loss
        total_step+=1
        iou=total_intersection/total_union
        pbar.set_postfix({'iou': iou,'iou_mean':np.mean(ious),'loss':total_loss/total_step})
        
    total_loss /= total_step
    iou = total_intersection/total_union
    return iou,total_loss


# In[ ]:





# In[18]:


fold0_all_mask=['aege', 'aidy', 'ajbl', 'ctjb', 'ctqb', 'ctui', 'ctzl',
       'ctzs', 'cubm', 'cuvz', 'cvdz', 'cvkl', 'hxhh', 'hxzu', 'hybs',
       'hyqa', 'hzrh', 'iasg', 'ibak', 'ibep', 'ibok', 'ibtu', 'rjvc',
       'wtep', 'wvcz', 'ysgy']

fold0_lim_mask_5=['aftk', 'agrp', 'aivi', 'csbd', 'cucs', 'cutq', 'cvcm', 'cvfq',
                   'cvhu', 'cvlg', 'cwbs', 'cwes', 'cwgw', 'hxzl', 'hyfv',
                   'hynp', 'hyou', 'hyxq', 'hyzo', 'hzbk', 'hzhw', 'hzoe', 'hzrq',
                   'hztg', 'hzza', 'ibcg', 'ibjo', 'ibld', 'ibot', 'ibpw', 'ibqa',
                   'ndjb', 'qpkk', 'qpwn', 'qpxg', 'qqbe', 'qqrp', 'qrif', 'qrpl',
                   'qrxa', 'qsag', 'qsaz', 'qtiv', 'wvzh', 'wwpd', 'ycgc', 'csxg', 
                  'cuqj', 'hxdf','hxfs', 'hygu', 'hyoe', 'hzdl', 'hzju', 'hznl',
                  'hzrj', 'ialc', 'iaoz', 'iave', 'iayu', 'ibck', 'ibmq', 'ibtq','jalp']

fold0_zero_mask=['ahfi', 'aisf', 'cutk', 'hxcm', 'hxdd', 'hxec', 'hxhj', 'hxlf',
                   'hxlo', 'hxlr', 'hxno', 'hxqm', 'hxqn', 'hxrn', 'hxrz', 'hxwk',
                   'hxxp', 'hxyc', 'hxyh', 'hxzc', 'hxzh', 'hxzp', 'hyae', 'hycr',
                   'hycu', 'hydv', 'hyei', 'hyej', 'hyfc', 'hyji', 'hylt', 'hynv',
                   'hyqb', 'hyrk', 'hysa', 'hzao', 'hzbr', 'hzcr', 'hzgu', 'hzip',
                   'hzsu', 'hzta', 'hztm', 'hzwe', 'hzxh', 'hzzt', 'iacc', 'iacl',
                   'iafn', 'iajb', 'iall', 'iamf', 'iamy', 'ianb', 'iapn', 'iasj',
                   'iasn', 'iasx', 'iauq', 'iayx', 'ibdb', 'ibek', 'ibeu', 'ibhj',
                   'ibhw', 'ibjw', 'ibkt', 'iblr', 'ibmj', 'ibmv', 'ibnb', 'iboj',
                   'ibou', 'iboy', 'ibpg', 'ibpy', 'ibre', 'ibro', 'ibry', 'ibsj',
                   'ibsk', 'ibtp', 'ibub', 'jxwa', 'mpbf', 'mqku', 'mqqi', 'mqxg',
                   'nnnt', 'pdjl', 'pelr', 'pfmm', 'qqcn', 'qrrb', 'wswi',
                   'wtdv', 'wtjn', 'wubp', 'ycro', 'ydou', 'yeuc']

fold1_all_mask=['bwoh', 'cnjv', 'coje', 'cqdc',
               'crae', 'dapl', 'dbam', 'dbdi', 'dbmi', 'dbtb', 'dbyp',
               'dcbg', 'dcbp', 'dchn', 'dchp', 'dcuw', 'dcvf', 'ddrm',
                'ddwm', 'ggfv', 'gjoe',
               'khey', 'khgv', 'nzxf', 'qixr', 'qlmn', 'wgsf',
               'ykzu']

fold1_lim_mask_5= ['auza', 'clvy', 'cmdp', 'coig',
                   'cpod', 'crjn', 'dcmm', 'ddpj', 'denk', 'dmig',
                   'dmjj', 'doww', 'dpmj', 'ggyg', 'kgqy', 'loar', 'lpnq',
                   'nzxy', 'oagv', 'qjbd', 'xltr', 'ynbb', 'ynbr','bvjo', 'bvss', 
                   'bxts', 'cndj', 'crfn','gjlr', 'kgqd','kgrz',
                   'kgwg', 'kgxy', 'khdw', 'kheh', '', 'khms','nzpm',
                   'ocnl', 'pluc','qjwj', 'yldz']

fold1_zero_mask = [ 'cnzy', 'coks','cozi', 'cpam', 'cpgd', 'cpsm',
               'crvm', 'dfwk', 'kgpp', 'kgpt', 'kgpz',
               'kgrm', 'kgsm', 'kgvd', 'kgvr', 'kgxp', 'khbj', 'khbs', 'khcp',
               'khlv', 'khmy', 'khof', 'lpnf',
               'lqjb', 'lqng', 'lqzx', 'lrax', 'mvmt', 'qiab', 'xvql','ynih']



fold2_all_mask = ['bhwy', 'bjmx', 'bknh', 'bkuu', 'dget', 'dhbr', 'dheq',
                  'djnl', 'djpf', 'dkox', 'dksz', 'ihob', 'iibg','ijka', 'ilel',
                  'kswl',  'kuvj','ohhv', 'ohkq', 'vzvt','wdzg', 'xrce', 'xrdt', 
                  'xrje', 'xrqy','xrzl', 'xsjc', 'xsqu', 'xteo', 'xtgh', 'xtkb', 
                  'xtnq', 'xtqb','xtsz', 'xtyt', 'xuei', 'xumv', 'xuvt', 'xuyt', 
                  'xvbz', 'xvcq','xvfq', 'xvnl', 'zvho', 'zwev','zxgd', 'zxwv']

fold2_lim_mask_5 =[ 'bgdq', 'bhbv','biyb', 'dgog',
                    'dgrk','ihva', 'jepz', 'ksmj',
                    'ohhi', 'ohka','ohnn','wank', 'wboo','wdgl','zvue',
                    'zwdz', 'zwnd', 'zwre','zxpp']

fold2_zero_mask = ['akkb', 'alfq', 'alht', 'apiw', 'asaq',
                   'bhzs', 'bjri', 'dhzm',
                   'djor', 'ekiw', 'emih', 'emnp', 'enhp', 'eojj', 
                   'jcle',
                   'jcmq', 'jdzi', 'ktfs', 'szqd', 'vpzw', 'xtwd', 'zwpu']

fold3_all_mask = [ 'ldla', 'phhx', 'phio', 'pjal', 'pjme', 'pjvf','sbfy', 'wxir', 'yxzm']

fold3_lim_mask_5 =['bsub', 'bvem', 'duse','ewbv', 'kzwv', 'kzym', 'lacv',
                   'lbsz', 'lcan', 'lchb', 'lckd', 'lcqm', 'lcst', 'lcvx',
                   'ldny', 'ldso', 'phyr','pibw', 'piwc', 'pjul', 'qycb', 
                   'qynn', 'qzac','qzhu', 'qziq', 'qzpj', 'rbcb', 'rccg', 
                   'rvou', 'rwcz','rwgq', 'yhah', 'yifd', 'yvlq', 'zaxb',
                   'zbli', 'zbpo']

fold3_zero_mask = [ 'duix', 'dxue', 'fmop', 'foxl','lbds', 'ldtr', 'phmo',
                   'phzr', 'pinj', 'pisl', 'qyfb', 'rbuz','susw', 'svpj', 
                   'wyhb', 'wzkj', 'xbpl','xgrz', 'yifm', 'yykc']

fold4_all_mask = [ 'gqgj', 'grci', 'grcs', 'grwn', 'gsfk', 'gshb', 'gspm', 'gtil',
                   'gtjz', 'gtve', 'gugh', 'hrkf', 'kyki', 'rrgq', 'rtfc',
                   'shcm', 'snal', 'vfak']

fold4_lim_mask_5 =['ezyv','gpve','gpve','gqwk','grdd','grmn','grpm','gsad','gsbn',
                   'gsdo','gsgc','gskf','gslr','gspj','gssh','gsvw','gszw','gtcy',
                   'gtjx','gtpg','gttb','gtuh','guhm','iwns','iyqh','izmo','kxzq',
                   'rthz','snwv','snyw','tfpt','uqnc' ,'uraz' ,'urnp' ,'utdl','utry',
                   'vdcl','yjtu','yjuv','yjyu','yjzw','ykaq','ykbl','ykdn', 'ykfd',
                   'ykgg','yhhu','ykib','ykid']

fold4_zero_mask = ['cdug', 'eyjs', 'eylb', 'eyrd', 'gsks', 'hsfa',
                   'izjh', 'kxgj', 'luml', 'smpm', 'snvl',
                    'uqvg', 'usjl', 'utts', 'vdat', 'vehl', 'ykfb']


# In[19]:


fold0_ign_mask = []
fold0_ign_mask.extend(fold0_all_mask)
fold0_ign_mask.extend(fold0_lim_mask_5)
fold0_ign_mask.extend(fold0_zero_mask)

fold1_ign_mask = []
fold1_ign_mask.extend(fold1_all_mask)
fold1_ign_mask.extend(fold1_lim_mask_5)
fold1_ign_mask.extend(fold1_zero_mask)

fold2_ign_mask = []
fold2_ign_mask.extend(fold2_all_mask)
fold2_ign_mask.extend(fold2_lim_mask_5)
fold2_ign_mask.extend(fold2_zero_mask)

fold3_ign_mask = []
fold3_ign_mask.extend(fold3_all_mask)
fold3_ign_mask.extend(fold3_lim_mask_5)
fold3_ign_mask.extend(fold3_zero_mask)

fold4_ign_mask = []
fold4_ign_mask.extend(fold4_all_mask)
fold4_ign_mask.extend(fold4_lim_mask_5)
fold4_ign_mask.extend(fold4_zero_mask)

len(fold0_ign_mask),len(fold1_ign_mask),len(fold2_ign_mask),len(fold3_ign_mask),len(fold4_ign_mask)


# In[20]:


GPU=True
def training_loop(data_path,params,train_x,val_x,savedir='./',mdl_name='resnet34'):
    
    #create model
    model = getModel(params)
    #load model
    #get loaders
    train_transforms=training_transformations
    val_transforms = None
    trainDataLoader,valDataLoader = getDataLoader(data_path,params,train_x,val_x,train_transforms,val_transforms)
    
    optimizer,scheduler,isStepScheduler = getOptimzersScheduler(model,params,
                                                                steps_in_epoch=len(trainDataLoader),
                                                                pct_start=0.1)
    best_iou = 0
    #control loop
    for e in range(params['max_epochs']):
        train_iou,train_loss = train_epoch(model,trainDataLoader,optimizer,scheduler,isStepScheduler)
        val_iou,val_loss = val_epoch(model,valDataLoader)
        #logging here
        print(e,'Train Result',f'loss={train_loss} iou={train_iou}')
        print(e,'Val Result',f'loss={val_loss} iou={val_iou}')
        if val_iou > best_iou :
            print(f'Saving for iou {val_iou}')
            save_model(e,model,ckpt_path=savedir,name=mdl_name)
            best_iou=val_iou
        else:
            print(f'Not Saving for iou {val_iou}')
        


# In[21]:


scaler = amp.GradScaler()


# In[22]:


import albumentations as A
training_transformations = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.0625,rotate_limit=15,p=0.5),
        A.GridDistortion(p=0.35),
        A.RandomCrop(384,384,p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(p=0.25),
    ]
)


# In[23]:




import random
seed=42
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    random.seed(0)
    np.random.seed(0)
set_seed(seed)


# In[27]:




def main(data_path,fold_num,num_worker):
    from sklearn.model_selection import GroupKFold
    
    import gc
    
    hparams = {
        # Optional hparams
        "backbone": 'timm-efficientnet-b1',
        "weights": "noisy-student",
        "learning_rate": 1e-3,
        "max_epochs": 25,
        "batch_size": 24,
        "num_workers": num_worker,
        "gpu": torch.cuda.is_available(),
        'div_factor':100,
        'final_div_factor':100,
    }

    train_metadata = pd.read_csv(f'{data_path}/train_metadata.csv')
    random.seed(9)  # set a seed for reproducibility

    to_exclude_data = [fold0_ign_mask,fold1_ign_mask,fold2_ign_mask,fold3_ign_mask,fold4_ign_mask]
    rem_all_fold_data = []
    for cids in to_exclude_data:
        rem_all_fold_data.extend(cids)

    train_metadata = train_metadata[~train_metadata.chip_id.isin(rem_all_fold_data)].copy().reset_index(drop=True)
    
    group_kfold = GroupKFold(n_splits=5)
    fold_df = train_metadata[['chip_id','location']].copy()
    fn=0
    for train_index, test_index in group_kfold.split(fold_df.chip_id, fold_df.chip_id, fold_df.location):
        fold_df.loc[test_index,'fold']=fn
        fn+=1

    fold_df.to_csv('fold_df.csv',index=False)
    
    fold_df = pd.read_csv('./fold_df.csv')
    train_metadata = train_metadata.merge(fold_df,on=['chip_id','location'])

    cloud_pct = pd.read_csv('./cloud_pct.csv')
    train_metadata = train_metadata.merge(cloud_pct,on=['chip_id'])
    
    
    version='1_caug'
    for fn in [fold_num]:  
        set_seed()
        mdl_name=hparams['backbone']
        savedir = '../models/'
        Path(savedir).mkdir(exist_ok=True, parents=True)

        val = train_metadata[train_metadata.fold==fn].copy().reset_index(drop=True)
        train = train_metadata[train_metadata.fold!=fn].copy().reset_index(drop=True)

        print('fold:',fn,'Train',train.shape,'Val',val.shape)

        training_loop(data_path,hparams,train,val,savedir=savedir,mdl_name=f'{mdl_name}-fold{fn}')
        gc.collect()

        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Segformer-b1 training')
    parser.add_argument('--data_dir', metavar='path', required=True,
                        help='the path to competition data')
    
    parser.add_argument('--fold_num', metavar='fold', required=True,
                        help='fold to train', type=int)
    
    parser.add_argument('--num_worker', nargs='?', default=0, type=int)
    args = parser.parse_args()
    
    print('args.num_worker',args.num_worker)
    
    main(data_path=args.data_dir,fold_num=args.fold_num,num_worker=args.num_worker)


# In[ ]:




