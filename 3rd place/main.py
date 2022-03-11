from pathlib import Path

from os import path, makedirs, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from tqdm import tqdm

from models import EfficientNet_Timm_Unet, Resnet_Timm_Unet, Timm_Unet

from Dataset import TestDataset

from PIL import Image


ROOT_DIRECTORY = Path("/codeexecution")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"


def main():
    makedirs(PREDICTIONS_DIRECTORY, exist_ok=True)

    cudnn.benchmark = True

    test_batch_size = 4

    test_files = []
    for d in sorted(listdir(INPUT_IMAGES_DIRECTORY)):
        if path.isdir(path.join(INPUT_IMAGES_DIRECTORY, d)):
            test_files.append(d)
    test_files = np.asarray(test_files)

    test_data = TestDataset(INPUT_IMAGES_DIRECTORY, test_files, bands = ['B02','B03','B04','B08'])

    test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=2, shuffle=False)


    models_4b_scaled = []
    for fold in [2]:

        model = EfficientNet_Timm_Unet(name='tf_efficientnet_b3_ns', pretrained=None, inp_size=4).cuda()
        snap_to_load = 'b3_4bands_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_4b_scaled.append(model)

        model = Timm_Unet(name='tf_efficientnetv2_s', pretrained=None, inp_size=4).cuda()
        snap_to_load = 'tf_efficientnetv2_s_4b_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_4b_scaled.append(model)


    models_3b_scaled = []
    for fold in [2]:
        
        model = Timm_Unet(name='tf_efficientnetv2_b0', pretrained=None, inp_size=3).cuda()
        snap_to_load = 'tf_efficientnetv2_b0_3b_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_3b_scaled.append(model)

        model = Resnet_Timm_Unet(name='resnet34', pretrained=None, inp_size=3).cuda()
        snap_to_load = 'res34_3b_scaled_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_3b_scaled.append(model)


    models_3b_minmax = []
    for fold in [2]:

        model = Resnet_Timm_Unet(name='resnet34', pretrained=False, inp_size=3).cuda()
        snap_to_load = 'res34_3b_minmax_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_3b_minmax.append(model)


    models_4b_minmax = []
    for fold in [2]:

        model = Timm_Unet(name='tf_efficientnetv2_b0', pretrained=None, inp_size=4).cuda()
        snap_to_load = 'tf_efficientnetv2_b0_4b_minmax_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_4b_minmax.append(model)


    models_4b = []
    for fold in [2]:

        model = EfficientNet_Timm_Unet(name='tf_efficientnet_b0_ns', pretrained=None, inp_size=4).cuda()
        snap_to_load = 'b0_4bands_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_4b.append(model)


    models_4b_2 = []
    for fold in [2]:

        model = Resnet_Timm_Unet(name='resnet34', pretrained=None, inp_size=4).cuda()
        snap_to_load = 'res34_pretrained_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(ASSETS_DIRECTORY, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval()
        models_4b_2.append(model)



    with torch.no_grad():
        for sample in tqdm(test_data_loader):
            chip_id = sample['chip_id']

            imgs_3b_minmax = sample['img_3b_minmax'].cpu().numpy()
            imgs_4b_minmax = sample['img_4b_minmax'].cpu().numpy()
            imgs_4b = sample['img_4b'].cpu().numpy()
            imgs_4b_2 = sample['img_4b_2'].cpu().numpy()
            imgs_4b_scaled = sample['img_4b_scaled'].cpu().numpy()
            imgs_3b_scaled = sample['img_3b_scaled'].cpu().numpy()

            msk_preds = []
            ids = []
            for i in range(0, len(chip_id), 1):
                ids.append(chip_id[i])
                msk_preds.append(np.zeros((512, 512), dtype='float'))

            cnt = 0
                
            for _tta in range(2):
                _i = _tta // 2
                _flip = False
                if _tta % 2 == 1:
                    _flip = True


                if _i == 0:
                    inp = imgs_3b_scaled.copy()
                elif _i == 1:
                    inp = np.rot90(imgs_3b_scaled, k=1, axes=(2,3)).copy()
                elif _i == 2:
                    inp = np.rot90(imgs_3b_scaled, k=2, axes=(2,3)).copy()
                elif _i == 3:
                    inp = np.rot90(imgs_3b_scaled, k=3, axes=(2,3)).copy()

                if _flip:
                    inp = inp[:, :, :, ::-1].copy()

                inp = torch.from_numpy(inp).float().cuda()                   
                
                for model in models_3b_scaled:
                    out = model(inp)
                    msk_pred = torch.sigmoid(out).cpu().numpy()

                    if _flip:
                        msk_pred = msk_pred[:, :, :, ::-1].copy()

                    if _i == 1:
                        msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                    elif _i == 2:
                        msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                    elif _i == 3:
                        msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                    cnt += 1

                    for i in range(len(ids)):
                        msk_preds[i] += msk_pred[i, 0]


                if _i == 0:
                    inp = imgs_4b_scaled.copy()
                elif _i == 1:
                    inp = np.rot90(imgs_4b_scaled, k=1, axes=(2,3)).copy()
                elif _i == 2:
                    inp = np.rot90(imgs_4b_scaled, k=2, axes=(2,3)).copy()
                elif _i == 3:
                    inp = np.rot90(imgs_4b_scaled, k=3, axes=(2,3)).copy()

                if _flip:
                    inp = inp[:, :, :, ::-1].copy()

                inp = torch.from_numpy(inp).float().cuda()                   
                
                for model in models_4b_scaled:
                    out = model(inp)
                    msk_pred = torch.sigmoid(out).cpu().numpy()

                    if _flip:
                        msk_pred = msk_pred[:, :, :, ::-1].copy()

                    if _i == 1:
                        msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                    elif _i == 2:
                        msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                    elif _i == 3:
                        msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                    cnt += 1

                    for i in range(len(ids)):
                        msk_preds[i] += msk_pred[i, 0]



                if _i == 0:
                    inp = imgs_3b_minmax.copy()
                elif _i == 1:
                    inp = np.rot90(imgs_3b_minmax, k=1, axes=(2,3)).copy()
                elif _i == 2:
                    inp = np.rot90(imgs_3b_minmax, k=2, axes=(2,3)).copy()
                elif _i == 3:
                    inp = np.rot90(imgs_3b_minmax, k=3, axes=(2,3)).copy()

                if _flip:
                    inp = inp[:, :, :, ::-1].copy()

                inp = torch.from_numpy(inp).float().cuda()                   
                
                for model in models_3b_minmax:
                    out = model(inp)
                    msk_pred = torch.sigmoid(out).cpu().numpy()

                    if _flip:
                        msk_pred = msk_pred[:, :, :, ::-1].copy()

                    if _i == 1:
                        msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                    elif _i == 2:
                        msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                    elif _i == 3:
                        msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                    cnt += 1

                    for i in range(len(ids)):
                        msk_preds[i] += msk_pred[i, 0]



                if _i == 0:
                    inp = imgs_4b_minmax.copy()
                elif _i == 1:
                    inp = np.rot90(imgs_4b_minmax, k=1, axes=(2,3)).copy()
                elif _i == 2:
                    inp = np.rot90(imgs_4b_minmax, k=2, axes=(2,3)).copy()
                elif _i == 3:
                    inp = np.rot90(imgs_4b_minmax, k=3, axes=(2,3)).copy()

                if _flip:
                    inp = inp[:, :, :, ::-1].copy()

                inp = torch.from_numpy(inp).float().cuda()                   
                
                for model in models_4b_minmax:
                    out = model(inp)
                    msk_pred = torch.sigmoid(out).cpu().numpy()

                    if _flip:
                        msk_pred = msk_pred[:, :, :, ::-1].copy()

                    if _i == 1:
                        msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                    elif _i == 2:
                        msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                    elif _i == 3:
                        msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                    cnt += 1

                    for i in range(len(ids)):
                        msk_preds[i] += msk_pred[i, 0]




                if _i == 0:
                    inp = imgs_4b.copy()
                elif _i == 1:
                    inp = np.rot90(imgs_4b, k=1, axes=(2,3)).copy()
                elif _i == 2:
                    inp = np.rot90(imgs_4b, k=2, axes=(2,3)).copy()
                elif _i == 3:
                    inp = np.rot90(imgs_4b, k=3, axes=(2,3)).copy()

                if _flip:
                    inp = inp[:, :, :, ::-1].copy()

                inp = torch.from_numpy(inp).float().cuda()                   
                
                for model in models_4b:
                    out = model(inp)
                    msk_pred = torch.sigmoid(out).cpu().numpy()

                    if _flip:
                        msk_pred = msk_pred[:, :, :, ::-1].copy()

                    if _i == 1:
                        msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                    elif _i == 2:
                        msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                    elif _i == 3:
                        msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                    cnt += 1

                    for i in range(len(ids)):
                        msk_preds[i] += msk_pred[i, 0]


                if _i == 0:
                    inp = imgs_4b_2.copy()
                elif _i == 1:
                    inp = np.rot90(imgs_4b_2, k=1, axes=(2,3)).copy()
                elif _i == 2:
                    inp = np.rot90(imgs_4b_2, k=2, axes=(2,3)).copy()
                elif _i == 3:
                    inp = np.rot90(imgs_4b_2, k=3, axes=(2,3)).copy()

                if _flip:
                    inp = inp[:, :, :, ::-1].copy()

                inp = torch.from_numpy(inp).float().cuda()                   
                
                for model in models_4b_2:
                    out = model(inp)
                    msk_pred = torch.sigmoid(out).cpu().numpy()

                    if _flip:
                        msk_pred = msk_pred[:, :, :, ::-1].copy()

                    if _i == 1:
                        msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                    elif _i == 2:
                        msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                    elif _i == 3:
                        msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                    cnt += 1

                    for i in range(len(ids)):
                        msk_preds[i] += msk_pred[i, 0]


            for i in range(len(ids)):
                msk_pred = msk_preds[i] / cnt

                _pred = msk_pred > 0.5
                
                _pred = _pred.astype('uint8')

                chip_pred_im = Image.fromarray(_pred)
                chip_pred_im.save(path.join(PREDICTIONS_DIRECTORY, '{}.tif'.format(ids[i])))


    print('OK')

if __name__ == "__main__":
    main()
