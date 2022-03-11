[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/cloud-cover-banner.jpg)

# On Cloud N: Cloud Cover Detection Challenge

## Goal of the Competition

Satellite imagery is critical for a wide variety of applications from disaster management and recovery, to agriculture, to military intelligence. Clouds present a major obstacle for all of these use cases, and usually have to be identified and removed from a dataset before satellite imagery can be used. Improving methods of identifying clouds can unlock the potential of an unlimited range of satellite imagery use cases, enabling faster, more efficient, and more accurate image-based research.

In this challenge, participants used machine learning to better detect cloud cover in satellite imagery from the [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) mission. Sentinel-2 captures wide-swath, high-resolution, multi-spectral images from around the world, and is publicly available through Microsoft's [Planetary Computer](https://planetarycomputer.microsoft.com/).

## What's in this Repository

This repository contains code from winning competitors in the [On Cloud N: Cloud Cover Detection](https://www.drivendata.org/competitions/83/cloud-cover/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Model scores represent a metric called [Jaccard index](https://www.drivendata.org/competitions/83/cloud-cover/page/398/#performance-metric), also known as Intersection Over Union (IoU).

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | [adityakumarsinha](https://www.drivendata.org/users/adityakumarsinha/) | 0.8988 | 0.8974 | An ensemble of two UNET model backbones was trained on significantly augmented images, including a custom augmentation for images with low cloud cover. Each model is trained separately on multiple folds of the data, generating 8 model weights whose predictions are averaged.
2   | Team IG NB: [XCZhou](https://www.drivendata.org/users/XCZhou/), [cloud_killer](https://www.drivendata.org/users/cloud_killer/), [Sunyueming](https://www.drivendata.org/users/Sunyueming/), [mymoon](https://www.drivendata.org/users/mymoon/), [windandcloud](https://www.drivendata.org/users/windandcloud/) | 0.8986 | 0.8972 | The data is cleaned and then randomly divided into training and validation sets five times. A base UNet++ model is trained on each random split, and the best-performing model is selected for integration. TTA is performed on the trained model with efficientnetv2_rw_s as an encoder during testing to increase the generalizability.
3   | [Victor](https://www.drivendata.org/users/Victor/) | 0.83975 | 0.8961 | The data is split into train/validation by location. 8 Unet-like models with different encoders are trained on the best single fold and then ensembled. Jaccard index is used as a metric, but calculated for each location separately and then averaged. Images are augmented before training and at test time.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: ["How to use deep learning, PyTorch lightning, and the Planetary Computer to predict cloud cover in satellite imagery"](https://www.drivendata.co/blog/cloud-cover-benchmark/)**
