source activate torch1.7
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_DeepLabV3Plus_timm_efb3_fold2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_UnetPlusPlus_timm_efb0_fold2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_UnetPlusPlus_timm_efb1_fold2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_UnetPlusPlus_timm_efb3_fold2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_UnetPlusPlus_timm_efb3_fold3
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_UnetPlusPlus_timm_efb5_fold2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_UnetPlusPlus_timm_efv2_rws_fold2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c train_UnetPlusPlus_timm_efv2_rws_fold3
