# CloudDetection Competition

We use the Unet++ model and the DeepLabV3+ model trained on five different distributions of training data and take the average of their outputs:

```
train_DeepLabV3Plus_timm_efb3_fold2
train_UnetPlusPlus_timm_efb0_fold2
train_UnetPlusPlus_timm_efb1_fold2
train_UnetPlusPlus_timm_efb3_fold2
train_UnetPlusPlus_timm_efb3_fold3
train_UnetPlusPlus_timm_efb5_fold2
train_UnetPlusPlus_timm_efv2_rws_fold2
train_UnetPlusPlus_timm_efv2_rws_fold3
```

## Setup

Linux ubantu18.0 + CUDA10.1 

```
●CPU (model): Intel(R) Xeon(R) CPU E5-2683 v3 @ 2.00GHz
●GPU (model or N/A): Nvidia Titan Xp*4
●Memory (GB):12GB*4
●OS: Linux ubantu18.0
●Train duration: training one Unet++ model takes ~ 6-9 hours, one Deeplabv3+ model takes ~ hours10.
●Inference duration: ~2.6 hours.
```


Create an environment using Python 3.7

```
conda create -n torch1.7 python=3.7
```

Install torch

```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
```

Install the required Python packages:

```
pip install -r requirements.txt
```


## Repo organization

```
──cloud_det
    |──checkpoint 		# storage location of trained models
    |──config 			# model configration
    |──dataset.py       # load dataset and data transform
    |──loss.py          # define loss function		
    |──models.py		# define models
    |──readme.md
    |──requirements.txt
    |──train.py			
    |──train.sh			# main function 
    |──utils.py			# metrics and gpu_data_augmentation
    |──data 			
         |──clear 		# image and label list for train and valid
         |    |──train_label.txt
         |    └──val_label.txt
         |──train_features
         |    └──train_chip_id_1
         |    |    |──B02.tif
         |    |    |──B03.tif
         |    |    |──B04.tif
         |    |    └─ B08.tif
         └──train_labels
              ├── train_chip_id_1.tif
    		  └── ...
    └──infer			# metrics of models on test dataset
         |──asset
         |    |──trained_model.pkl
         |    └── ...
         |──dataset.py
         |──main.py
         └──models
```

First, we iteratively cleaned the original dataset to remove about 1000 images that we thought would cause the model to lose accuracy.
Since we used a multi-fold training method, both the training labels and the validation labels are split into five here.

## training and inference

### Run training

Single GPU：

exapmle: 

if you want to train deeplabv3+ ,

```
python train.py -c train_DeepLabV3Plus_timm_efb3_fold3
```

DataParallel train:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py -c train_DeepLabV3Plus_timm_efb3_fold3
```

'train_DeepLabV3Plus_timm_efb3_fold3' is the parameter configuration file in the 'config' folder

### Run inference

```
cd infer
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

## Model configuration

The parameter configuration file located in the 'config' folder

```
config = dict(
    seed=10000,
    data_path="./data",
    Kfold_index=2,#0,1,2,3,4
    train_image_id_txt_path='',
    val_image_id_txt_path='',
    encoder='efficientnet-b3',
    model_network='DeepLabV3Plus',
    in_channels=4,
    n_class=1,
    save_path='',
    gpu_da=0.25,
    max_epochs = 50,
    train_batch_size = 16,
    test_batch_size = 16,
    lr = 0.0003,
    weight_decay = 0.0005,
    save_inter_epoch = 5,
    print_freq = 50,
    num_workers = 8,
)
config['save_path']='{}_{}_fold_{}'.format(config['encoder'],config['model_network'],config['Kfold_index'])
config['train_image_id_txt_path']=config['data_path']+"/clear/train_label_five_fold_remove_noisy_{}.txt".format(config['Kfold_index'])
config['val_image_id_txt_path']=config['data_path']+"/clear/val_label_five_fold_remove_noisy_{}.txt".format(config['Kfold_index'])
```

seed: Fix all random seeds during training
data_path: dataset root path
Kfold_index: Multi-fold cross-validation selection 
train_image_id_txt_path: image id of the training set
val_image_id_txt_path: image id of the validation set
encoder: backbone of the model
model_network: network of the model
in_channels：input channles of the model
n_class：output class number of the model
save_path：Log file and model parameter file storage path
gpu_da：data augmentation probability on gpu
max_epochs：Maximum number of training epochs
train_batch_size：training batch size
test_batch_size：Validated batch size

