import os
from pathlib import Path
from typing import List
from loguru import logger
import numpy as np
import pandas as pd
from tifffile import imwrite
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import typer


from segformer import SegFormer_b1,AmpNet
from cloud_model import CloudDataset,Net4CH
import torchvision

ROOT_DIRECTORY = Path("/codeexecution")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"


# Make sure the smp loader can find our torch assets because we don't have internet!
os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "torch")
GPU=torch.cuda.is_available()
print("GPU Available",GPU)


def get_metadata(features_dir: os.PathLike, bands: List[str]):
    """
    Given a folder of feature data, return a dataframe where the index is the chip id
    and there is a column for the path to each band's TIF image.

    Args:
        features_dir (os.PathLike): path to the directory of feature data, which should have
            a folder for each chip
        bands (list[str]): list of bands provided for each chip
    """
    chip_metadata = pd.DataFrame(index=[f"{band}_path" for band in bands])
    chip_ids = (
        pth.name for pth in features_dir.iterdir() if not pth.name.startswith(".")
    )

    for chip_id in chip_ids:
        chip_bands = [features_dir / chip_id / f"{band}.tif" for band in bands]
        chip_metadata[chip_id] = chip_bands

    return chip_metadata.transpose().reset_index().rename(columns={"index": "chip_id"})

def getModel():
    model = AmpNet()
    model.cuda()
    return model

def getModel4ch(hparams):
    unet_model = Net4CH(hparams)
    unet_model.cuda()
    return unet_model


def prediction_step(data, models,th=0.5,predictions_dir='./'):
    is_mixed_precision = True
   
    chip_ids = data['chip_id']
    images = data['chip'].float()
    images = images.cuda()

    preds = np.zeros((images.shape[0],images.shape[2],images.shape[3]))
    
    images = torch.stack([images, torchvision.transforms.functional.hflip(images),
                            torchvision.transforms.functional.vflip(images)], 0)
    n, bs, c, h, w = images.size()
    images = images.view(-1, c, h, w)
    
    #print('prediction_step',preds.shape,images.shape)
    for model in models:
        model.eval()
        with torch.no_grad():
            mask = model(images)
            
            probs1, probs2, probs3 = torch.split(mask, bs)
            probs2 = torchvision.transforms.functional.hflip(probs2)
            probs3 = torchvision.transforms.functional.vflip(probs3)
            
            mask =  (1/3)*probs1 + (1/3)*probs2 + (1/3)*probs3
            mask = mask.sigmoid()
            preds += mask[:,0].cpu().numpy()
    preds /= len(models)
    preds = ((preds>0.5)*1).astype(np.uint8)
    
    for ix in range(len(chip_ids)):
        chip_id = chip_ids[ix]
        output_path = predictions_dir / f"{chip_id}.tif"
        #if ix == 0:
        #    print('prediction_step',chip_id,preds[ix].shape,(preds[ix]==1).sum())
        imwrite(output_path, preds[ix], dtype=np.uint8)
        


hparams = {
    "backbone": 'timm-efficientnet-b1',
    "weights": "noisy-student",
}

is_mixed_precision = True
import gc
def main(
    assets_dir_path: Path = ASSETS_DIRECTORY,
    test_features_dir: Path = DATA_DIRECTORY / "test_features",
    predictions_dir: Path = PREDICTIONS_DIRECTORY,
    bands: List[str] = ["B02", "B03", "B04"],):

    if not test_features_dir.exists():
        raise ValueError(
            f"The directory for test feature images must exist and {test_features_dir} does not exist"
        )
    predictions_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("Loading model")
    # Explicitly set where we expect smp to load the saved resnet from just to be sure
    torch.hub.set_dir(assets_dir_path / "torch/hub")
    
    
    models = []
    
    assets_dir_path1 = assets_dir_path/'segformer-b1'
    model_paths = assets_dir_path1.glob('*.pth')
    
    for mp in model_paths:
        print('Loading Model from Path',mp)
        f = torch.load(mp, map_location=torch.device('cpu'))
        model = getModel()
        model.load_state_dict(f)
        model.cuda()
        models.append(model)
        del f
        gc.collect()
        
    assets_dir_path2 = assets_dir_path/'eff1_4ch'
    model_paths = assets_dir_path2.glob('*.pth')
    
    for mp in model_paths:
        print('Loading Model from Path',mp)
        f = torch.load(mp, map_location=torch.device('cpu'))
        model = getModel4ch(hparams)
        model.load_state_dict(f)
        model.cuda()
        models.append(model)
        del f
        gc.collect()
        
    torch.cuda.empty_cache()
    
    logger.info("Finding chip IDs")
    chip_id_metadata = get_metadata(test_features_dir, bands)
    
    logger.info(f"Found {len(chip_id_metadata)} test chip_ids. Generating predictions.")
    
    #create dataset and data loader
    testDataSet = val_dataset = CloudDataset(chip_id_metadata, 
                                             x_path= test_features_dir, 
                                             bands = [4,3,2,8])
    testDataLoader = torch.utils.data.DataLoader(
                        testDataSet,
                        batch_size=16,num_workers=4,shuffle=False,pin_memory=False
                        )
    
    for i,data in tqdm(enumerate(testDataLoader),total=len(testDataLoader)):
        prediction_step(data,models,predictions_dir=predictions_dir)

    logger.success(f"Inference complete.")


if __name__ == "__main__":
    typer.run(main)
