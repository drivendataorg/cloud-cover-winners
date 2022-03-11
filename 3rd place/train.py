import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs
import numpy as np

from contextlib import suppress

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed

from torch.optim import SGD

from losses import iou_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from models import EfficientNet_Timm_Unet, Resnet_Timm_Unet, Timm_Unet

from Dataset import TrainDataset
from utils import *

from timm.utils.distributed import distribute_bn

from ddp_utils import all_gather, reduce_tensor

from torch.utils.tensorboard import SummaryWriter

# import warnings
# warnings.filterwarnings("ignore")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amp', default=True, type=bool)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--encoder", default='tf_efficientnetv2_s')
parser.add_argument("--checkpoint", default='tf_efficientnetv2_s_4b')
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--minmax', default=False, type=bool)
parser.add_argument('--bands', default=4, type=int)
parser.add_argument('--norm_max', default=-1, type=int)
parser.add_argument('--epoches', default=80, type=int)
parser.add_argument("--checkpoint_path", default='')

args, unknown = parser.parse_known_args()

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

local_rank = 0
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
args.local_rank = local_rank



df = pd.read_csv('folds.csv')

data_dir = 'data'

models_folder = 'assets'
    
chip2loc = df.groupby('chip_id')['location'].first().to_dict()



def validate(model, val_data_loader, current_epoch, amp_autocast=suppress):
    metrics = [[] for i in range(4)]
    ids = []

    if args.local_rank == 0:
        iterator = tqdm(val_data_loader)
    else:
        iterator = val_data_loader

    with torch.no_grad():
        for i, sample in enumerate(iterator):
            with amp_autocast():
                chip_ids = sample["chip_id"]
                locs = sample["location"]
                imgs = sample["img"].cuda(non_blocking=True)
                otps = sample["msk"].cpu().numpy()

                res = model(imgs)

                probs = torch.sigmoid(res)
                pred = probs.cpu().numpy()
                
                for j in range(otps.shape[0]):
                    ids.append(chip_ids[j])
                    
                    _truth = otps[j, 0]
                    _pred = pred[j, 0]
                    if _truth.shape[0] != 512:
                        _truth = cv2.resize(_truth.astype('float32'), (512, 512))
                        _pred = cv2.resize(_pred.astype('float32'), (512, 512))
                    _pred = _pred > 0.5
                    _truth = _truth > 0.5

                    _int = np.logical_and(_truth, _pred).sum()
                    _un = np.logical_or(_truth, _pred).sum()
                    _iou = 0
                    if _un > 0:
                        _iou = _int / _un
                    metrics[0].append(_iou)
                    metrics[1].append(_int)
                    metrics[2].append(_un)

                    

    metrics = [np.asarray(x) for x in metrics]
    ids = np.asarray(ids)

    if args.distributed:
        metrics = [np.concatenate(all_gather(x)) for x in metrics]
        ids = np.concatenate(all_gather(ids))
        torch.cuda.synchronize()

    _iou_mean = np.mean(metrics[0])
    _int = metrics[1].sum()
    _un = metrics[2].sum()

    _iou = 0
    if _un > 0:
        _iou = _int / _un
    

    by_loc = {}
    for i in range(len(ids)):
        l = chip2loc[ids[i]]
        if l not in by_loc:
            by_loc[l] = ([], [], [])
        by_loc[l][0].append(metrics[0][i])
        by_loc[l][1].append(metrics[1][i])
        by_loc[l][2].append(metrics[2][i])

    _locs = list(by_loc.keys())
    _sc_weighted = 0
    _iou_weighted = 0
    for i in range(len(_locs)):
        l = _locs[i]
        _iou0 = np.mean(by_loc[l][0])
        _int = np.sum(by_loc[l][1])
        _un = np.sum(by_loc[l][2])

        _sc0 = 0
        if _un > 0:
            _sc0 = _int / _un

        _sc_weighted += _sc0
        _iou_weighted += _iou0

    _sc_weighted /= len(_locs)
    _iou_weighted /= len(_locs)

    if args.local_rank == 0:
        print("Val Total IOU_all: {} iou_mean: {} IOU_all_w: {} iou_mean_w: {} Len: {}".format(_iou, _iou_mean, _sc_weighted, _iou_weighted, len(metrics[0])))


        writer.add_scalar("IOU/Val", _iou, current_epoch)
        writer.add_scalar("IOU_mean/Val", _iou_mean, current_epoch)
        writer.add_scalar("IOU_w/Val", _sc_weighted, current_epoch)
        writer.add_scalar("IOU_mean_w/Val", _iou_weighted, current_epoch)

    return _sc_weighted




def evaluate_val(val_data_loader, best_score, model, snapshot_name, current_epoch, amp_autocast=suppress):
    model.eval()
    _sc = validate(model, val_data_loader, current_epoch, amp_autocast)

    if args.local_rank == 0:
        if _sc > best_score:
            if args.distributed:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_score': _sc,
                }, path.join(models_folder, snapshot_name))
            else:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': _sc,
                }, path.join(models_folder, snapshot_name))

            best_score = _sc
        print("Val score: {}\tbest_score: {}".format(_sc, best_score))
    return best_score, _sc



def train_epoch(current_epoch, combo_loss, model, optimizer, scaler, train_data_loader, amp_autocast=suppress):
    losses = [AverageMeter() for i in range(10)]
    metrics = [AverageMeter() for i in range(10)]
    
    if args.local_rank == 0:
        iterator = tqdm(train_data_loader)
    else:
        iterator = train_data_loader

    _lr = optimizer.param_groups[0]['lr']

    model.train()

    for i, sample in enumerate(iterator):
        with amp_autocast():
            imgs = sample["img"].cuda(non_blocking=True)
            otps = sample["msk"].cuda(non_blocking=True)

            res = model(imgs)

            loss = combo_loss(res, otps)

        _dices = []
        with torch.no_grad():
            for _i in range(1):
                _probs = torch.sigmoid(res[:, _i, ...])
                dice_sc = 1 - iou_round(_probs, otps[:, _i, ...])
                _dices.append(dice_sc)
            del _probs

        if args.distributed:
            reduced_loss = [reduce_tensor(x.data) for x in [loss]]
            reduced_sc = [reduce_tensor(x) for x in _dices]
        else:
            reduced_loss = [x.data for x in [loss]]
            reduced_sc = _dices

        for _i in range(len(reduced_loss)):
            losses[_i].update(to_python_float(reduced_loss[_i]), imgs.size(0))
        for _i in range(len(reduced_sc)):
            metrics[_i].update(reduced_sc[_i], imgs.size(0)) 

        if args.local_rank == 0:
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {:.4f} ({:.4f}); iou: {:.4f} ({:.4f})".format(
                    current_epoch, _lr, losses[0].val, losses[0].avg, metrics[0].val, metrics[0].avg))


        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.999)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.999)
            optimizer.step()

        torch.cuda.synchronize()

    if args.local_rank == 0:
        writer.add_scalar("Loss/train", losses[0].avg, current_epoch)
        writer.add_scalar("IOU/train", metrics[0].avg, current_epoch)
        writer.add_scalar("lr", _lr, current_epoch)

        print("epoch: {}; lr {:.7f}; Loss {:.4f}; iou: {:.4f};".format(
                    current_epoch, _lr, losses[0].avg, metrics[0].avg))


            

if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    makedirs(models_folder, exist_ok=True)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0 
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()


    fold = args.fold


    if args.local_rank == 0:
        writer = SummaryWriter(comment='{}_{}'.format(args.checkpoint, fold))
        print(args)
        
    
    cudnn.benchmark = True

    batch_size = args.batch_size
    val_batch = args.batch_size

    best_snapshot_name = '{}_{}_best'.format(args.checkpoint, fold)
    last_snapshot_name = '{}_{}_last'.format(args.checkpoint, fold)

    df_train = df[df['fold'] != fold].copy()
    df_train = df_train.reset_index(drop=True)
    df_val = df[df['fold'] == fold].copy()
    df_val = df_val.reset_index(drop=True)

    new_size = None
    norm_max = -1
    bands = ['B02', 'B03', 'B04', 'B08']
    bands = bands[:args.bands]

    data_train = TrainDataset(df=df_train, data_dir=data_dir, aug=True, bands=bands, norm_max=args.norm_max, new_size=new_size, minmax=args.minmax)
    data_val = TrainDataset(df=df_val, data_dir=data_dir, aug=False, bands=bands, norm_max=args.norm_max, new_size=new_size, minmax=args.minmax)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_val)


    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=4, shuffle=(train_sampler is None), pin_memory=False, sampler=train_sampler)
    val_data_loader = DataLoader(data_val, batch_size=val_batch, num_workers=4, shuffle=False, pin_memory=False, sampler=val_sampler)


    if 'tf_efficientnetv2' in args.encoder:
        model = Timm_Unet(name=args.encoder, pretrained=args.pretrained, inp_size=len(bands), checkpoint_path=args.checkpoint_path)
    elif 'resne' in args.encoder:
        model = Resnet_Timm_Unet(name=args.encoder, pretrained=args.pretrained, inp_size=len(bands), checkpoint_path=args.checkpoint_path)
    else:
        model = EfficientNet_Timm_Unet(name=args.encoder, pretrained=args.pretrained, inp_size=len(bands), checkpoint_path=args.checkpoint_path)
    

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()

    params = model.parameters()
    
    optimizer = SGD(params, lr=2e-3, momentum=0.9)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
            output_device=args.local_rank)


    loss_scaler = None
    amp_autocast = suppress
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
        amp_autocast = torch.cuda.amp.autocast

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=200, verbose=False, threshold=0.00001, threshold_mode='abs', cooldown=0, min_lr=1e-06, eps=1e-06)

    combo_loss = ComboLoss({'jaccard': 1.0, 'focal': 0.1}, per_image=True).cuda() #


    best_score = 0
    for epoch in range(args.epoches):
        torch.cuda.empty_cache()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(epoch, combo_loss, model, optimizer, loss_scaler, train_data_loader, amp_autocast)

        if args.distributed:
            distribute_bn(model, args.world_size, True)

        best_score, _sc = evaluate_val(val_data_loader, best_score, model, best_snapshot_name, epoch, amp_autocast)
        scheduler.step(_sc)
        
        
        if args.local_rank == 0:
            writer.flush()

            # save last?
            # if args.distributed:
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'state_dict': model.module.state_dict(),
            #         'best_score': best_score,
            #     }, path.join(models_folder, last_snapshot_name + '_' + str(epoch)))
            # else:
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'state_dict': model.state_dict(),
            #         'best_score': best_score,
            #     }, path.join(models_folder, last_snapshot_name + '_' + str(epoch)))
    
    torch.cuda.empty_cache()
    if args.distributed:
        torch.cuda.synchronize()

    del model

    elapsed = timeit.default_timer() - t0
    if args.local_rank == 0:
        writer.close()
        print('Time: {:.3f} min'.format(elapsed / 60))