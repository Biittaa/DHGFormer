from pathlib import Path
import argparse
import yaml
import torch
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import logging

from train import BasicTrain
from model.DHGFormer import DHGFormer
from kfold_dataloader import init_dataloader_kfold
from util import Logger_main


def main(args, config, fold_idx, kfold, base_seed):
    current_seed = base_seed + fold_idx
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)

    val_ratio = config['train'].get('val_ratio', config.get('data', {}).get('val_set', 0.1))

    # dataloaders, node_size, node_feature_size, timeseries_size, smri_dim = \
    #     init_dataloader_kfold(
    #         config['data'],
    #         fold_idx=fold_idx,
    #         kfold=kfold,
    #         val_ratio=val_ratio,
    #         seed=base_seed
    #     )

    # config['train']["seq_len"] = timeseries_size
    # config['train']["node_size"] = node_size

    # model = DHGFormer(config['model'], node_size,
    #                  node_feature_size, timeseries_size,
    #                  use_smri=config['data'].get('use_smri', False),
    #                  smri_input_dim=smri_dim)
    
    dataloaders, node_size, node_feature_size, timeseries_size, smri_dim, \
        mvgcn_view_meta, mvgcn_fold_graphs = \
        init_dataloader_kfold(
            config['data'],
            fold_idx=fold_idx,
            kfold=kfold,
            val_ratio=val_ratio,
            seed=base_seed
        )

    config['train']["seq_len"] = timeseries_size
    config['train']["node_size"] = node_size

    # smri_encoder_type فقط زیر data: تنظیم می‌شه؛ اینجا به model: هم منتقل
    # می‌شه که DHGFormer.__init__ بدون دوباره‌نویسی توی هر دو بخش، همون مقدار رو بخونه.
    config['model']['smri_encoder_type'] = config['data'].get('smri_encoder_type', 'fcn')

    model = DHGFormer(config['model'], node_size,
                     node_feature_size, timeseries_size,
                     use_smri=config['data'].get('use_smri', False),
                     smri_input_dim=smri_dim,
                     mvgcn_view_meta=mvgcn_view_meta,
                     mvgcn_fold_graphs=mvgcn_fold_graphs)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'])
    opts = (optimizer,)

    save_folder_name = Path(config['train']['log_folder']) / Path(config['model']['type']) / Path(
        f"{config['data']['dataset']}_{config['data']['atlas']}") / Path(f"fold_{fold_idx+1}")

    train_process = BasicTrain(
        config['train'], model, opts, dataloaders, save_folder_name)

    # train_process.train()
    return train_process.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_DHGFormer.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--kfold', default=5, type=int,
                        help='Number of folds for stratified k-fold cross validation.')
    parser.add_argument('--seed', default=21, type=int)
    parser.add_argument('--device', default=4, type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    logger = Logger_main()

    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    kfold = config['train'].get('kfold', args.kfold)

    logger.info(f"Model {config['model']['type']} on {config['data']['dataset']} Dataset")
    logger.info(f"Running real Stratified {kfold}-Fold Cross Validation, base seed:{seed}")

    all_fold_metrics = []
    for fold_idx in range(kfold):
        logger.info(f"Fold {fold_idx + 1}/{kfold}, base_seed:{seed}, device:{args.device}")
        with open(args.config_filename) as f:
            config = yaml.load(f, Loader=yaml.Loader)  # fresh copy per fold
        fold_metrics = main(args, config, fold_idx, kfold, seed)
        all_fold_metrics.append(fold_metrics)
        logger.info(f"Fold {fold_idx + 1} is done!")
        
        
    logger.info("==== K-Fold Summary (mean ± std) ====")
    for name in all_fold_metrics[0].keys():
        values = np.array([m[name] for m in all_fold_metrics])
        logger.info(f"{name}: {values.mean():.4f} ± {values.std():.4f}")

    logging.info(f"Done!")