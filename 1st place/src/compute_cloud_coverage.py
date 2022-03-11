#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm


# In[3]:


def main(data_dir):
    train = pd.read_csv(f'{data_dir}/train_metadata.csv')
    lbl_sum = []
    for chip_id in tqdm(train.chip_id.values,total=train.shape[0]):
        lpth = f'{data_dir}/train_labels/{chip_id}.tif'
        with rasterio.open(lpth) as lp:
            y = lp.read(1).astype(int)
            lbl_sum.append(y[y!=255].sum())
    train['lbl_sz'] = np.array(lbl_sum)/(512*512)
    train[['chip_id','lbl_sz']].to_csv('cloud_pct.csv',index=False)


# In[4]:


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute Clou Coverage')
    parser.add_argument('--data_dir', metavar='path', required=True,
                        help='the path to competition data')
    args = parser.parse_args()
    main(data_dir=args.data_dir)


# In[ ]:




