# Cloud Cover Detection Challenge, Solution
## Conda Environment
*Execute following command to create conda env with name condaenv_cloud using  environement file environment-gpu.yml. The file is same as available at following location https://github.com/drivendataorg/cloud-cover-runtime/blob/main/runtime/environment-gpu.yml*

```
conda env create -f environment-gpu.yml
conda activate condaenv_cloud
pip install transformers==4.12.0
```
```
cd src
```

### Execute Following command to generate cloud coverage ratio of labels ans save it in cloud_pct.csv file
```
python compute_cloud_coverage.py --data_dir <competition data directory>
```
**The library is released on Oct 28,2021 as per https://pypi.org/project/transformers/4.12.0/**
## Training for best Submission
*Go to src directory to execute following commands*
>*Note: For downloading Pretrain weights for timm-efficientnet-b1 and  nvidia/segformer-b1-finetuned-ade-512-512, internet access is required.*


### Execute following commmand to train UNET with backbaone as timm-efficientnet-b1 backbone

**The code will generate model weights in model directory with name timm-efficientnet-b1-fold<0-4>.pth**

```
python unet_efficient_net_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 0 --num_worker 6
python unet_efficient_net_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 1 --num_worker 6
python unet_efficient_net_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 2 --num_worker 6
python unet_efficient_net_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 3 --num_worker 6
python unet_efficient_net_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 4 --num_worker 6
```
  
*In final submission model weights for all fold are used.*

### Execute following commmand to train SegformerForSemanticSegmentation model as available in HuggingFace transformers pretrain model 'nvidia/segformer-b1-finetuned-ade-512-512'
**The code will generate model weights in model directory with name segformer_b1-fold<0-4>.pth**

```
python segformer_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 0 --num_worker 6
python segformer_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 1 --num_worker 6
python segformer_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 2 --num_worker 6
python segformer_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 3 --num_worker 6
python segformer_b1_train_fp16_custom_aug.py --data_dir <competition data directory> --fold_num 4 --num_worker 6
```
 
*In final submission model weights for fold 0,2,4 are used.*

## Training for third best submission 
### Execute following commmand to train UNET with backbaone as tf_efficientnetv2_b2 backbone
**The code will generate model weights in model directory with name tf_efficientnetv2_b2-fold<0-4>.pth**

```
python unet_efficient_netV2_b2_train_fp16_custom_aug.py --data_dir <data directory> --fold_num 0 --num_worker 6
python unet_efficient_netV2_b2_train_fp16_custom_aug.py --data_dir <data directory> --fold_num 1 --num_worker 6
python unet_efficient_netV2_b2_train_fp16_custom_aug.py --data_dir <data directory> --fold_num 2 --num_worker 6
python unet_efficient_netV2_b2_train_fp16_custom_aug.py --data_dir <data directory> --fold_num 3 --num_worker 6
python unet_efficient_netV2_b2_train_fp16_custom_aug.py --data_dir <data directory> --fold_num 4 --num_worker 6
```
  
*In final submission model weights for all fold are used.*

### Execute following commmand to train UNET with backbaone as timm-efficientnet-b1 backbone
**The code will generate model weights in model directory with name timm-efficientnet-b1-fold<0-4>_wo_ca.pth**
*This code is not using custom augmentation*

```
python unet_efficient_net_b1_train_fp16.py --data_dir <data directory> --fold_num 0 --num_worker 6
python unet_efficient_net_b1_train_fp16.py --data_dir <data directory> --fold_num 1 --num_worker 6
python unet_efficient_net_b1_train_fp16.py --data_dir <data directory> --fold_num 2 --num_worker 6
python unet_efficient_net_b1_train_fp16.py --data_dir <data directory> --fold_num 3 --num_worker 6
python unet_efficient_net_b1_train_fp16.py --data_dir <data directory> --fold_num 4 --num_worker 6
```  
*In final submission model weights for all fold are used.*


## Submission for best Submission
### Model Weights is available at following link

Folder SegFromer-b1-cutom-aug has SegformerForSemanticSegmentation-b1 weights
Folder has UNET-efficientnet-b1-custom-aug has unet_efficientnet_b1 weights
https://drive.google.com/drive/folders/14sVfEhVFeoNdGdXG3BQXsUb4kI1OiUBd?usp=sharing

**Execute following notebook to final submission**
  
**Notebook with name segformer_b1_unet_effnet_b1_ensemble_v1.ipynb is used for final submission. In this notebook SegformerForSemanticSegmentation is copy pasted (Not a clean solution)**
  
**Alternatively submission can be generated by segformer_b1_unet_effnet_b1_ensemble_v1_with_transformers notebook and it is directly importing SegformerForSemanticSegmentation from transformers library**
  
*The execution of notebook will create directory with named segformer_b1_unet_effnet_b1_ensemble_submission. The directory has neecessary structure as required for competition.*
>*Note: For downloading Pretrain weights for timm-efficientnet-b1 and  nvidia/segformer-b1-finetuned-ade-512-512, internet access is required.*


## Submission for third best Submission
### Model Weights is available at following link

Folder SegFromer-b1-cutom-aug has SegformerForSe
Folder has UNET-efficientnetV2-b2 unet_effnetv2b2 weights
Folder has UNET-efficientnet-b1 unet_effnet_b1 weights
https://drive.google.com/drive/folders/14sVfEhVFeoNdGdXG3BQXsUb4kI1OiUBd?usp=sharing

**Execute following notebook to final submission**
  
**Notebook with name ens_effnetv2b2_unet_eff1.ipynb is used for submission.** 
  
*The execution of notebook will create directory with named effnetv2b2_unet_eff1_ensemble. The directory has neecessary structure as required for competition.*
  
>*Note: For downloading Pretrain weights for timm-efficientnet-b1 and  tf_efficientnetv2_b2, internet access is required.*
