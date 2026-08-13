
import numpy as np
import torch
import torch.utils.data as utils
import csv

from nilearn.connectome import ConnectivityMeasure
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from nilearn import plotting, datasets
import random

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_subject_order(order_path):
    order_map = {}
    with open(order_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.lower().startswith('index_in_drive'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            subj = str(int(parts[1]))
            order_map[idx] = subj
    return order_map

def load_smri_features(smri_path):
    df = pd.read_csv(smri_path)

    def to_numeric_id(raw_id):
        tail = str(raw_id).split('_')[-1]
        return str(int(tail))

    df['subject_num'] = df['subject_id'].apply(to_numeric_id)

    feature_cols = [c for c in df.columns if c not in ('subject_id', 'subject_num')]
    feat_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    feat_df = feat_df.fillna(feat_df.mean()).fillna(0.0)

    mean = feat_df.mean(axis=0).values
    std = feat_df.std(axis=0).values
    std[std == 0] = 1.0
    normed = (feat_df.values - mean) / std

    subject_to_feat = {sid: normed[i] for i, sid in enumerate(df['subject_num'].values)}
    return subject_to_feat, normed.shape[1]

def build_smri_tensor(dataset_config, num_subjects):
    order_map = load_subject_order(dataset_config['time_series_subjects_order'])
    subject_to_feat, smri_dim = load_smri_features(dataset_config['smri'])

    smri_matrix = np.zeros((num_subjects, smri_dim), dtype=np.float32)
    missing = 0
    for i in range(num_subjects):
        subj = order_map.get(i)
        if subj is not None and subj in subject_to_feat:
            smri_matrix[i] = subject_to_feat[subj]
        else:
            missing += 1
    if missing > 0:
        print(f'[sMRI] Warning: sMRI data not found for {missing}/{num_subjects} subjects. Zeros were assigned.')
    return torch.from_numpy(smri_matrix).float(), smri_dim


        
def init_dataloader(dataset_config):
    data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
    final_fc = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]


    _, _, timeseries = final_fc.shape

    _, node_size, node_feature_size = final_pearson.shape

    scaler = StandardScaler(mean=np.mean(
        final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc)


    pseudo = []
    for i in range(len(final_fc)):
        pseudo.append(np.diag(np.ones(final_pearson.shape[1])))

    if 'cc200' in  dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 200, 200))
    elif 'aal' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 116, 116))
    elif 'cc400' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 392, 392))
    else:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))


    final_fc, final_pearson, labels, pseudo_arr = [torch.from_numpy(
        data).float() for data in (final_fc, final_pearson, labels, pseudo_arr)]
    length = final_fc.shape[0]
    
    use_smri = dataset_config.get('use_smri', False)
    if use_smri:
        smri_tensor, smri_size = build_smri_tensor(dataset_config, length)
    else:
        smri_tensor = torch.zeros((length, 1), dtype=torch.float32)
        smri_size = 0
    
    train_length = int(length*dataset_config["train_set"])
    val_length = int(length*dataset_config["val_set"])

    
    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels,
        pseudo_arr,
        smri_tensor
    )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length, length-train_length-val_length])

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    
    return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries, smri_size
