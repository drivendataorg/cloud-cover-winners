from pathlib import Path
from dataset import LoadTifDataset
import numpy as np
from tifffile import imsave
import torch
from models import CloudModel
import time
from time import strftime, localtime


def load_model_state_dict(checkpoint_path):
    checkpoints = torch.load(checkpoint_path)
    assert checkpoint_path.split(".")[-1] in ['pth', 'pkl']
    if checkpoint_path.split(".")[-1] == 'pth':
        return checkpoints['state_dict']
    else:
        return checkpoints


if __name__ == "__main__":
    time_start = time.time()
    print(strftime('%Y-%m-%d %H:%M:%S', localtime()))
    batch_size=32# resnet50,1_5fold,gpu,6245M;efb1,1_5fold,gpu,4173M;
    num_workers=12
    #data path
    root_dir = Path("/codeexecution")
    #model'Fast_SCNN',
    networks = [
        'DeepLabV3Plus',
        'UnetPlusPlus', 'UnetPlusPlus',
        'UnetPlusPlus', 'UnetPlusPlus', 'UnetPlusPlus', 'UnetPlusPlus', 'UnetPlusPlus']
    encoders = [
        'efficientnet-b3',
        'timm-efficientnet-b0', 'timm-efficientnet-b1',
        'timm-efficientnet-b3', 'timm-efficientnet-b3','timm-efficientnet-b5',
        'tu-efficientnetv2_rw_s','tu-efficientnetv2_rw_s']
    checkpoint_paths = [
        './asset/efficientnet-b3_DeepLabV3Plus_reference_the_top1_solution_3_fold_clear.pkl',
        "./asset/timm-efficientnet-b0_UnetPlusPlus_reference_the_top1_solution_5_fold_clear.pkl",
        "./asset/timm-efficientnet-b1_UnetPlusPlus_reference_the_top1_solution_2_fold_clear.pkl",
        "./asset/timm-efficientnet-b3_UnetPlusPlus_reference_the_top1_solution_4_fold_clear.pkl",
        "./asset/timm-efficientnet-b3_UnetPlusPlus_reference_the_top1_solution_5_fold_clear.pkl",
        "./asset/timm-efficientnet-b5_UnetPlusPlus_reference_the_top1_solution_5_fold_clear.pkl",
        "./asset/tu-efficientnetv2_rw_s_UnetPlusPlus_reference_the_top1_solution_2_fold_clear.pkl",
        "./asset/tu-efficientnetv2_rw_s_UnetPlusPlus_reference_the_top1_solution_5_fold_clear.pkl",   
    ]
    models = []
    in_channels = 4
    n_class = 1
    for i in range(len(networks)):
        network = networks[i]
        encoder = encoders[i]
        checkpoint_path = checkpoint_paths[i]
        model = CloudModel(encoder=encoder, network=network,
                           in_channels=in_channels, n_class=n_class,pre_train=None).cuda()
        model = torch.nn.DataParallel(model)
        checkpoints = load_model_state_dict(checkpoint_path)
        model.load_state_dict(checkpoints)
        models.append(model)
    #pred path
    PREDICTIONS_DIRECTORY = root_dir / "predictions"
    INPUT_IMAGES_DIRECTORY = root_dir / "data/test_features"
    chip_ids = [pth.name for pth in INPUT_IMAGES_DIRECTORY.iterdir() if not pth.name.startswith(".")]
    test_dataset=LoadTifDataset(img_dir=INPUT_IMAGES_DIRECTORY,chip_ids=chip_ids)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            data = batch["chip"].cuda()# [batchn,c,h,w]
            batch_out_all = []
            for idx, model in enumerate(models):
                model.eval()  # [b,c,h,w]
                if idx < 6: # 0 1 2 3 4 5
                    output = model(data)[:, 0, :, :]  # [b,c,h,w]->[b,1,h,w]->[b,h,w]
                    output = torch.sigmoid(output).cpu().numpy().astype('float32')
                else:
                    predict_1 = model(data)[:, 0, :, :]  # [b,c,h,w]->[b,1,h,w]->[b,h,w]
                    predict_2 = model(torch.flip(data, [-1]))[:, 0, :, :]  # [b,c,h,w]->[b,1,h,w]->[b,h,w]
                    predict_2 = torch.flip(predict_2, [-1])
                    predict_3 = model(torch.flip(data, [-2]))[:, 0, :, :]  # [b,c,h,w]->[b,1,h,w]->[b,h,w]
                    predict_3 = torch.flip(predict_3, [-2])
                    output = (torch.sigmoid(predict_1) + torch.sigmoid(predict_2)
                              + torch.sigmoid(predict_3)).cpu().numpy().astype('float32') / 3
                batch_out_all.append(output)
            output_mean = np.mean(batch_out_all, 0)  #[num_models,b,512,512]->[b,512,512],mean
            batch_pred = np.where(output_mean > 0.55, 1, 0).astype('uint8')#[b,512,512]
            for chip_id, pred in zip(batch["chip_id"], batch_pred):
                chip_pred_path = PREDICTIONS_DIRECTORY / f"{chip_id}.tif"
                imsave(chip_pred_path, pred)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')