import os
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
def train_transform():
    trained_transform = A.Compose([
        A.ToFloat(max_value=65536.0),
        A.RandomCrop(384,384),
        ToTensorV2(transpose_mask=True),
    ])
    return trained_transform
def val_transform():
    infer_transform = A.Compose([
        A.ToFloat(max_value=65536.0),
        ToTensorV2(),
    ])
    return infer_transform
class CloudDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform, img_id_txt_path=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_id_txt_path = img_id_txt_path
        self.Band_name=['B02','B03','B04','B08']
        self.ids = [id.strip() for id in open(self.img_id_txt_path) if len(id)>0]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        band_arrs = []
        for band in self.Band_name:
            band_path=os.path.join(self.img_dir, idx+'/'+band+'.tif')
            band_img=io.imread(band_path)
            band_arrs.append(band_img)#b,g,r,nir
        img = np.stack(band_arrs, axis=-1)
        label_path = os.path.join(self.label_dir, idx + '.tif')
        label = io.imread(label_path)
        sample = self.transform(image=img, mask=label)
        sample['mask'] = torch.unsqueeze(sample['mask'], dim=0)
        return sample['image'], sample['mask'].float()
