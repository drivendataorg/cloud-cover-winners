import random
from utils import *
import pandas as pd

df = pd.read_csv('train_metadata.csv')

df['fold'] = -1
num_folds = 4
_idxs = [i for i in range(num_folds)]

i = -1
for l in df['location'].value_counts().index.values:
    i += 1
    if i % num_folds == 0:
        random.shuffle(_idxs)
    df.loc[df['location'] == l, 'fold'] = _idxs[i % num_folds]
    

df.to_csv('folds.csv', index=False)