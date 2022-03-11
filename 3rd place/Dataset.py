import torch
from torch.utils.data import Dataset

import numpy as np
import random

import cv2

from imgaug import augmenters as iaa

from utils import *

from os import path

import rasterio

from pathlib import Path



band_max = {
    'B02': 22500,
    'B03': 21000,
    'B04': 19500,
    'B08': 17000,
    'B01': 24000,
    'B11': 15500,
    'SCL': 11,
}


class TrainDataset(Dataset):
    def __init__(self, df, data_dir, aug=True, bands=['B02','B03','B04','B08'], norm_max=-1, new_size=None, minmax=False, tune=False):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.aug = aug
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.bands = bands
        self.new_size = new_size
        self.norm_max = norm_max

        self.minmax = minmax
        self.img_dir = Path(path.join(data_dir, 'train_features'))
        self.tune = tune


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.loc[idx]

        img = []
        for band in self.bands:
            with rasterio.open(path.join(self.data_dir, 'train_features', row['chip_id'], '{}.tif'.format(band))) as b:
                band_arr = b.read(1).astype("float32")
            if self.minmax:
                band_arr = (band_arr - band_arr.min()) / (band_arr.max() - band_arr.min())
            else:
                if self.norm_max < 0:
                    band_arr = band_arr / band_max[band]
                    band_arr[band_arr > 1.0] = 1.0
                else:
                    band_arr = band_arr / self.norm_max
                    band_arr[band_arr > 1.0] = 1.0
            img.append(band_arr)

        img = np.stack(img, axis=-1)


        with rasterio.open(path.join(self.data_dir, 'train_labels', '{}.tif'.format(row['chip_id']))) as b:
            msk = b.read(1).astype("float32")
        msk = (msk * 255).astype("uint8")

        if self.tune and row['iou'] < 0.2:
            msk = cv2.imread(path.join('full_oof', row['chip_id'] + '.png'), flags=cv2.IMREAD_UNCHANGED)


        if self.aug:
            _p = 0.5
            if random.random() > _p:
                img = img[:, ::-1, :]
                msk = msk[:, ::-1]

            if random.random() > 0.0:
                _k = random.randrange(4)
                img = np.rot90(img, k=_k, axes=(0,1))
                msk = np.rot90(msk, k=_k, axes=(0,1))

            _p = 0.3
            if random.random() > _p:
                _d = int(img.shape[0] * 0.4)
                rot_pnt =  (img.shape[0] // 2 + random.randint(-_d, _d), img.shape[1] // 2 + random.randint(-_d, _d))
                scale = 1
                if random.random() > 0.2:
                    scale = random.normalvariate(1.0, 0.1)
                angle = 0
                if random.random() > 0.2:
                    angle = random.randint(0, 90) - 45
                if (angle != 0) or (scale != 1):
                    img = rotate_image(img, angle, scale, rot_pnt)
                    msk = rotate_image(msk, angle, scale, rot_pnt)

            if random.random() > 0.99:
                for j in range(img.shape[2]):
                    _d = random.random() * 0.1 - 0.05
                    img[:, :, j] = img[:, :, j] + _d
                img[img > 1.0] = 1.0
                img[img < 0] = 0

            if random.random() > 0.99:
                for j in range(img.shape[2]):
                    scale = random.normalvariate(1.0, 0.1)
                    img[:, :, j] = scale * img[:, :, j]
                img[img > 1.0] = 1.0


            if random.random() > 0.96:
                el_det = self.elastic.to_deterministic()
                _2byte = False
                for i in range(img.shape[2]):
                    img[:, :, i] = el_det.augment_image(img[:, :, i])


            if random.random() > 0.85:
                sz0 = random.randrange(1, int(img.shape[0] * 0.4))
                sz1 = random.randrange(1, int(img.shape[1] * 0.4))
                x0 = random.randrange(img.shape[1] - sz1)
                y0 = random.randrange(img.shape[0] - sz0)
                img[y0:y0+sz0, x0:x0+sz1, :] = 0
                msk[y0:y0+sz0, x0:x0+sz1] = 0



        if self.new_size is not None:
            img = cv2.resize(img, self.new_size)
            msk = cv2.resize(msk, self.new_size)

        msk = msk[:, :, np.newaxis]

        msk = (msk > 127)

        img = preprocess_inputs_float(img)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        sample = {'img': img, 'msk': msk, 'chip_id': row['chip_id'], 'location': row['location']}

        return sample


class TestDataset(Dataset):
    def __init__(self, data_dir, chip_ids, bands=['B02','B03','B04','B08'], new_size=None):
        super().__init__()
        self.chip_ids = chip_ids
        self.data_dir = data_dir
        self.bands = bands
        self.new_size = new_size
        

    def __len__(self):
        return len(self.chip_ids)


    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]


        img_4b_minmax = []
        img_4b_scaled = []
        for band in self.bands:
            with rasterio.open(path.join(self.data_dir, chip_id, '{}.tif'.format(band))) as b:
                band_arr = b.read(1).astype("float32")

            img_4b_minmax.append(band_arr.copy())

            band_arr = band_arr / band_max[band]
            band_arr[band_arr > 1.0] = 1.0
            img_4b_scaled.append(band_arr)


        img_4b = np.stack(img_4b_minmax, axis=-1)

        img_4b_minmax = img_4b.copy()

        img_4b_minmax = (img_4b_minmax - img_4b_minmax.min(axis=(0, 1))) / (img_4b_minmax.max(axis=(0, 1)) - img_4b_minmax.min(axis=(0, 1)))
        img_3b_minmax = img_4b_minmax[..., :3].copy()

        img_4b_2 = img_4b.copy() / 20000.0
        img_4b_2[img_4b_2 > 1.0] = 1.0

        img_4b = img_4b / 30000.0
        img_4b[img_4b > 1.0] = 1.0

        img_4b_scaled = np.stack(img_4b_scaled, axis=-1)
        img_3b_scaled = img_4b_scaled[..., :3].copy()


        img_4b_minmax = preprocess_inputs_float(img_4b_minmax)
        img_4b_minmax = torch.from_numpy(img_4b_minmax.transpose((2, 0, 1)).copy()).float()

        img_3b_minmax = preprocess_inputs_float(img_3b_minmax)
        img_3b_minmax = torch.from_numpy(img_3b_minmax.transpose((2, 0, 1)).copy()).float()
        
        img_4b = preprocess_inputs_float(img_4b)
        img_4b = torch.from_numpy(img_4b.transpose((2, 0, 1)).copy()).float()

        img_4b_2 = preprocess_inputs_float(img_4b_2)
        img_4b_2 = torch.from_numpy(img_4b_2.transpose((2, 0, 1)).copy()).float()

        img_4b_scaled = preprocess_inputs_float(img_4b_scaled)
        img_4b_scaled = torch.from_numpy(img_4b_scaled.transpose((2, 0, 1)).copy()).float()

        img_3b_scaled = preprocess_inputs_float(img_3b_scaled)
        img_3b_scaled = torch.from_numpy(img_3b_scaled.transpose((2, 0, 1)).copy()).float()

        sample = {'img_3b_minmax': img_3b_minmax, 'img_4b_minmax': img_4b_minmax, 'img_4b': img_4b, 'img_4b_2': img_4b_2, 'img_4b_scaled': img_4b_scaled, 'img_3b_scaled': img_3b_scaled, 'chip_id': chip_id}

        return sample