import argparse
import importlib
import os
import shutil
from utils import train_net
from utils import setup_seed
from dataset import CloudDataset
from dataset import train_transform,val_transform
from models import CloudModel
if __name__ == "__main__":
    # set random seed
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-c', '--config', type=str, help='Configuration File')
    config_name=parser.parse_args().config
    config = importlib.import_module("." + config_name, package='config').config
    setup_seed(config['seed'])
    # data path
    data_path = config['data_path']#'./data'
    img_dir_path = os.path.join(data_path, "train_features")
    label_dir_path = os.path.join(data_path, "train_labels")
    Kfold=config['Kfold_index']#0,1,2,3,4
    train_image_id_txt_path = config['train_image_id_txt_path']
    val_image_id_txt_path = config['val_image_id_txt_path']
    # dataset
    train_transform = train_transform()
    val_transform = val_transform()
    train_dataset = CloudDataset(img_dir=img_dir_path, label_dir=label_dir_path,
                                      img_id_txt_path=train_image_id_txt_path, transform=train_transform)
    valid_dataset = CloudDataset(img_dir=img_dir_path, label_dir=label_dir_path,img_id_txt_path=val_image_id_txt_path,
                                      transform=val_transform)
    # model
    model = CloudModel(encoder=config['encoder'], network=config['model_network'],
                     in_channels=config['in_channels'], n_class=config['n_class'])

    # model save path
    save_ckpt_path = os.path.join('./checkpoint', config['save_path'], 'ckpt')
    save_log_path = os.path.join('./checkpoint', config['save_path'])
    if not os.path.exists(save_ckpt_path):
        os.makedirs(save_ckpt_path)
    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)
    config['save_log_path'] = save_log_path
    old_config_name_path='./config'+'/'+config_name+'.py'
    new_config_name_path = config['save_log_path'] + '/' + config_name + '.py'
    shutil.copyfile(src=old_config_name_path,dst=new_config_name_path)
    #copy the config.py to the log path
    config['save_ckpt_path'] = save_ckpt_path
    # training
    train_net(config=config, model=model, train_dataset=train_dataset, valid_dataset=valid_dataset)
