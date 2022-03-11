import torch
import numpy as np
import os
import tifffile
class LoadTifDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        chip_ids,
    ):
        """Dataset for training, validating and testing S2 models.

        Args:
            img_paths (list of str): Paths to the input B02 path.
            mask_paths (list of str): Paths to the labels for the S2 images.
            transforms (albumentation.transforms, optional): Transforms to apply to the images/masks. Defaults to None.
            val (bool, optional): If True, this dataset is used for validation.
                Defaults to False.
            test (bool, optional): If True, we don't provide the label, because we are testing. Defaults to False.
        """
        self.img_dir = img_dir
        self.chip_ids = chip_ids
    def __len__(self):
        return len(self.chip_ids)
    def __getitem__(self, idx):
        chip_id=self.chip_ids[idx]
        sample = {}
        # Load in image
        #arr_x = tifffile.imread([self.img_dir,idx.replace("B02.tif", f"B0{k}.tif") for k in [2, 3, 4, 8]])
        arr_x = tifffile.imread([os.path.join(self.img_dir, chip_id+'/'+f"B0{k}.tif") for k in [2, 3, 4, 8]])
        #os.path.join(self.img_dir, idx+'/'+band+'.tif')
        arr_x = (arr_x / 2 ** 16).astype(np.float32)
        sample["chip"] = arr_x
        sample["chip_id"] = chip_id
        return sample