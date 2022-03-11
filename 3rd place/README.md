# On Cloud N: Cloud Cover Detection Challenge - 3rd Place Solution

Username: Victor Durnov

## Summary

- Final solution is the ensemble of 8 Unet-like models with different pretrained encoders (resnet34, efficientnetv2_b0, efficientnetv2_s, efficientnet_b0, efficientnet_b3) from [TIMM](https://github.com/rwightman/pytorch-image-models)
- 3-bands and 4-bands images with different preprocessing used as input (min-max normalization, thresholds for each band)
- Tried to use additional B01, B11 and SCL bands from Planetary Computer, but no improvements on both local CV and Leaderboard
- Using of larger encoders or additional bands tends to fast overfitting and worse results.
- Train/Validation split made by location evenly using number of samples. Refer to `create_folds.py` for details. Best single fold on public test set also best in final test set
- Modified metric used to select best checkpoint: Jaccard index calculated for each location separately and then averaged.
- Augmentations used: random flips, rotations, scale, cut-outs, elastic
- Training loss: `jaccard + 0.1 * focal` 
- Optimizer: SGD with `lr=2e-3, momentum=0.9`. Batch size of 4. Adam optimizer also leads to fast overfitting to training set
- No data or labels cleaning performed, this worsed results both on CV and leaderboard
- Horizontal test-time augmentation used on prediction


# Setup

Solution is dockerized. Refer to `Dockerfile` for details. 

To build docker image: 

```
 docker build -t cloud .
```

It will download trained models to `/codeexecution/assets` on container start.

To run the docker container use something like this:

```
 docker run --gpus all -v /<host_dir>:/<container_dir> --ipc=host -it cloud
```

Or if don't need to download models:

```
 docker run --gpus all -v /<host_dir>:/<container_dir> --ipc=host -it cloud /bin/bash
```

The working directory is `/codeexecution`
Before training copy data to `/codeexecution/data/train_features` and `/codeexecution/data/train_labels`
Before testing copy data to `/codeexecution/data/test_features`


# Hardware

The solution was trained on workstation with 4 Titan V gpus (12 GB), so training designed for similiar environment (otherwise change `nproc_per_node` parameter in `train.sh`)

Training time: ~1 day

Inference time: 3 hours on 1 GPU


# Run training

`./train.sh` script contains all required to train all models. Just need to copy data to `/codeexecution/data` folder.

Trained models will be saved (overwrited) to `/codeexecution/assets` folder. All training logs will be written to `/codeexecution/logs` 


# Run inference

`./test.sh` script will create predictions for all inputs from `/codeexecution/data/test_features`. With downloaded trained weights it should fullly reproduce winning submission.
Results saved to `/codeexecution/predictions` and logs written to `/codeexecution/logs`