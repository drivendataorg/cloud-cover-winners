import numpy as np
import logging
import random
import os
import time
from loss import XEDiceLoss
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def rotate(input, degrees=90):
    """(..., H, W) input expected"""
    if degrees == 90:
        return input.transpose(-2, -1).flip(-2)
    if degrees == 180:
        return input.flip(-2).flip(-1)
    if degrees == 270:
        return input.transpose(-2, -1).flip(-1)


def transpose(input):
    """(..., H, W) input expected"""
    return input.transpose(-2, -1)


def apply_rotate_transpose(input, rot90, rot180, rot270, transpose):
    transformed: torch.Tensor = input.clone()
    to_rot90 = rot90.to(input.device)
    transformed[to_rot90] = rotate(input[to_rot90], degrees=90)
    to_rot180 = rot180.to(input.device)
    transformed[to_rot180] = rotate(input[to_rot180], degrees=180)
    to_rot270 = rot270.to(input.device)
    transformed[to_rot270] = rotate(input[to_rot270], degrees=270)
    to_transpose = transpose.to(input.device)
    transformed[to_transpose] = transformed[to_transpose].transpose(-2, -1)
    return transformed


def gpu_da(x_data, y_data, gpu_da_params):
    with torch.no_grad():
        bs = y_data.size(0)

        no_dihedral_p = gpu_da_params
        transpose, rot90, rot180, rot270 = get_transpose_rot_boolean_lists(bs, no_dihedral_p)

        transpose, rot90, rot180, rot270 = (
            torch.tensor(transpose),
            torch.tensor(rot90),
            torch.tensor(rot180),
            torch.tensor(rot270),
        )
        debug_show = False
        if debug_show:
            raise NotImplementedError()
        else:
            x_data = apply_rotate_transpose(x_data, rot90, rot180, rot270, transpose)
            y_data = apply_rotate_transpose(y_data, rot90, rot180, rot270, transpose)
            return x_data, y_data


def get_transpose_rot_boolean_lists(bs, no_dihedral_p):
    """
    In no_dihedral_p % do nothing, in (1-no_dihedral_p) % / 7 do one of the 7 possible transpose/rot combinations.
    """
    transpose, rot90, rot180, rot270 = [False] * bs, [False] * bs, [False] * bs, [False] * bs
    perc_for_each_combination = (1 - no_dihedral_p) / 7
    for k in range(bs):
        rand_float = np.random.random()
        if rand_float < perc_for_each_combination:
            rot90[k] = True
        elif rand_float < 2 * perc_for_each_combination:
            rot180[k] = True
        elif rand_float < 3 * perc_for_each_combination:
            rot270[k] = True
        elif rand_float < 4 * perc_for_each_combination:
            rot90[k] = True
            transpose[k] = True
        elif rand_float < 5 * perc_for_each_combination:
            rot180[k] = True
            transpose[k] = True
        elif rand_float < 6 * perc_for_each_combination:
            rot270[k] = True
            transpose[k] = True
        elif rand_float < 7 * perc_for_each_combination:
            transpose[k] = True
        else:
            pass
    # print(f"transpose: {sum(transpose)}/{bs}, 90degree rot: {sum(rot90)}/{bs}, 180degree rot: {sum(rot180)}/{bs}, 270degree rot: {sum(rot270)}/{bs}")
    return transpose, rot90, rot180, rot270


def init_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

class Metrics:
    """
    Computes and stores segmentation related metrices for training
    """

    def __init__(self) -> None:
        self.tps, self.fps, self.fns, self.iou = 0, 0, 0, 0

    def update_metrics(self, preds, targets):
        tps, fps, fns = tp_fp_fn_with_ignore(preds, targets)
        self.tps += tps
        self.fps += fps
        self.fns += fns

    def calc_ious(self):
        """
        Calculates IoUs per class and biome, mean biome IoUs, penalty and final metric used for early stopping
        """
        self.iou = self.tps / (self.tps + self.fps + self.fns)
        self.early_stopping_metric = self.iou


def tp_fp_fn_with_ignore(preds, targets):
    """
    Calculates True Positives, False Positives and False Negatives ignoring pixels where the target is 255.

    Args:
        preds (float tensor): Prediction tensor
        targets (long tensor): Target tensor
        c_i (int, optional): Class value of target for the positive class. Defaults to 1.

    Returns:
        tps, fps, fns: True Positives, False Positives and False Negatives
    """
    preds = preds.flatten()
    targets = targets.flatten()

    # ignore missing label pixels
    no_ignore = targets.ne(255)
    preds = preds.masked_select(no_ignore)
    targets = targets.masked_select(no_ignore)

    # calculate TPs/FPs/FNs on all water
    tps = torch.sum(preds * (targets == 1))
    fps = torch.sum(preds) - tps
    fns = torch.sum(targets == 1) - tps

    return tps, fps, fns

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_net(config, model, train_dataset, valid_dataset):
    max_epochs = config['max_epochs']
    batch_size = config['train_batch_size']
    test_batch_size = config['test_batch_size']
    lr = config['lr']
    weight_decay = config['weight_decay']

    save_inter_epoch = config['save_inter_epoch']
    print_freq = config['print_freq']

    save_log_path = config['save_log_path']
    save_ckpt_path = config['save_ckpt_path']
    num_workers = config['num_workers']
    scaler = torch.cuda.amp.GradScaler()
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=test_batch_size, shuffle=False,num_workers=num_workers)
    # loss function
    loss_func = XEDiceLoss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-5
    )

    # logging
    logger = init_logger(
        os.path.join(save_log_path, time.strftime("%m-%d-%H-%M-%S", time.localtime()) + '.log'))

    train_loader_size = train_data_loader.__len__()
    epoch_init = 0
    best_metric =0
    model.cuda()
    model = DataParallel(model)
    logger.info('starting training,max epoch:{}'.format(max_epochs))
    for curr_epoch_num in range(epoch_init, max_epochs):
        start_time = time.time()
        # training
        model.train()
        losses, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
        gpu_da_time = AverageMeter()
        end = time.time()
        for iter_num, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            data, target = batch[0].cuda(), batch[1].cuda()
            start = time.time()
            if config['gpu_da'] != 0:
                data, target = gpu_da(data, target, config['gpu_da'])
            gpu_da_time.update(time.time() - start)
            with torch.cuda.amp.autocast():
                pred = model(data)
                loss = loss_func(pred, target)
            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            # Unscales the gradients, then optimizer.step() is called if gradients are not inf/nan,
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
            scheduler.step(curr_epoch_num + iter_num / train_loader_size)
            losses.update(loss.detach().item(), data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if iter_num > 0 and iter_num % print_freq == 0:
                logger.info(
                    'Ep:{} {}/{} total time {:.3f}min'.format(curr_epoch_num,iter_num,train_loader_size,(time.time() - start_time) / 60)
                )
                logger.info(
                    "BatchT: {:.3f}s, DataT: {:.3f}s, GpuDaT: {:.3f}s, Loss: {:.4f}".format(batch_time.avg,data_time.avg,gpu_da_time.avg,losses.avg))
                losses.reset()
        logger.info(
            f"Ep: [{curr_epoch_num}] TotalT: {(time.time() - start_time) / 60:.1f} min, "
            f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, GpuDaT: {gpu_da_time.avg:.3f}s, Loss: {losses.avg:.4f}"
        )
        # validation
        model.eval()

        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
        metrics = Metrics()

        end = time.time()
        with torch.no_grad():
            for iter_num, batch in enumerate(valid_data_loader):
                data, target = batch[0].cuda(), batch[1].cuda()
                pred = torch.where(torch.sigmoid(input=model(data)) > 0.5, 1, 0)
                metrics.update_metrics(pred.cpu(), target.cpu())
            batch_time.update(time.time() - end)
            metrics.calc_ious()
            # Log results
            logger.info(
                f"Ep: [{curr_epoch_num}]  ValT: {(batch_time.avg * len(valid_data_loader)) / 60:.2f} min, BatchT: {batch_time.avg:.3f}s, "
                f"DataT: {data_time.avg:.3f}s, IoU: {metrics.iou:.4f} (val)"
            )
        if curr_epoch_num % save_inter_epoch == 0:
            state = {'epoch': curr_epoch_num, 'best_metric': best_metric, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_path, 'checkpoint_epoch{}.pth'.format(curr_epoch_num))
            torch.save(state, filename)

        # Save the best model
        if metrics.iou > best_metric:
            best_metric = metrics.iou
            state = {'epoch': curr_epoch_num, 'best_metric': best_metric, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            logger.info('Best Iou Model saved at epoch:{}'.format(curr_epoch_num))
