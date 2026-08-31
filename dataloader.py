
import numpy as np
import torch
import torch.utils.data as utils
import csv
import re

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


def load_smri_features(dataset_config, num_subjects):
    """Load sMRI tabular features (abide_smri.csv style) and align them,
    subject by subject, with the fMRI tensors already loaded from abide.npy.
    Alignment uses time_series_subjects_order: one subject id per line, in the
    exact same order as axis 0 of abide.npy (timeseires/corr/label)."""
    order_path = dataset_config["time_series_subjects_order"]
    
    
    order_df = pd.read_csv(
        order_path,
        sep="\t",
        header=None,
        names=["index_in_drive", "subject_id", "site"],
        skiprows=2
    )

    order_df = order_df.dropna(subset=["subject_id"]).copy()
    
    
    # order_df["subject_id"] = (
    #     order_df["subject_id"]
    #     .astype(str)
    #     .str.strip()
    # )
    
    order_df["subject_id"] = (
        order_df["subject_id"]
        .astype(str)
        .str.strip()
        .apply(lambda x: str(int(x)))
    )
    
        
    order_df = order_df[
        order_df["subject_id"].str.fullmatch(r"\d+")
    ].copy()
    
    subject_order = order_df["subject_id"].tolist()


    print(f"Subject order file: {len(subject_order)} subjects")
    print(f"fMRI data:          {num_subjects} subjects")
    
    smri_df = pd.read_csv(dataset_config["smri_path"])
    
    # smri_df["SUB_ID"] = smri_df["subject_id"].apply(
    #     lambda s: re.findall(r"\d+", str(s))[-1]
    #     if re.findall(r"\d+", str(s))
    #     else None
    # )
    
    smri_df["SUB_ID"] = smri_df["subject_id"].apply(
        lambda s: str(int(re.findall(r"\d+", str(s))[-1]))
        if re.findall(r"\d+", str(s))
        else None
    )
    

    non_feature_cols = ["subject_id", "SUB_ID"]
    feature_cols = [c for c in smri_df.columns
                    if c not in non_feature_cols and pd.api.types.is_numeric_dtype(smri_df[c])]
    print(f"sMRI feature dimension: {len(feature_cols)}")
    smri_lookup = {
        row["SUB_ID"]: row[feature_cols].values.astype(np.float64)
        for _, row in smri_df.iterrows()
        if row["SUB_ID"] is not None
    }
    
    

    feature_dim = len(feature_cols)
    if len(subject_order) != num_subjects:
        raise ValueError(
            f"Subject-order file contains {len(subject_order)} subjects, "
            f"but fMRI contains {num_subjects} subjects.\n"
            f"Do NOT simply truncate the list because that can "
            f"misalign fMRI and sMRI subjects."
        )
    
    
    smri_features = np.full((num_subjects, feature_dim), np.nan, dtype=np.float64)

    missing_count = 0
    for i, sub_id in enumerate(subject_order):
    
        sub_id_clean = str(sub_id).strip()

        if sub_id_clean in smri_lookup:
            smri_features[i, :] = smri_lookup[sub_id_clean]
        else:
            missing_count += 1

    print(f"sMRI missing subjects: {missing_count}")
    
    col_means = np.nanmean(smri_features, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_rows, nan_cols = np.where(np.isnan(smri_features))
    smri_features[nan_rows, nan_cols] = col_means[nan_cols]
    
    feat_std = smri_features.std(axis=0)
    keep_cols = feat_std > 1e-8
    if not np.all(keep_cols):
        print(f"sMRI: dropping {np.sum(~keep_cols)} constant feature column(s)")
    
    smri_features = smri_features[:, keep_cols]
    feature_cols = [c for c, k in zip(feature_cols, keep_cols) if k]
    feature_dim = smri_features.shape[1]

    smri_scaler = StandardScaler(mean=np.mean(smri_features, axis=0),
                                  std=np.std(smri_features, axis=0) + 1e-8)
    smri_features = smri_scaler.transform(smri_features)

    return smri_features, feature_dim




        
# def init_dataloader(dataset_config):
#     data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
#     final_fc = data["timeseires"]
#     final_pearson = data["corr"]
#     labels = data["label"]


#     _, _, timeseries = final_fc.shape

#     _, node_size, node_feature_size = final_pearson.shape

#     scaler = StandardScaler(mean=np.mean(
#         final_fc), std=np.std(final_fc))
    
#     final_fc = scaler.transform(final_fc)


#     pseudo = []
#     for i in range(len(final_fc)):
#         pseudo.append(np.diag(np.ones(final_pearson.shape[1])))

#     if 'cc200' in  dataset_config['atlas']:
#         pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 200, 200))
#     elif 'aal' in dataset_config['atlas']:
#         pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 116, 116))
#     elif 'cc400' in dataset_config['atlas']:
#         pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 392, 392))
#     else:
#         pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))

#     use_smri = dataset_config.get("use_smri", False)
#     num_subjects = final_fc.shape[0]
#     if use_smri:
#         smri_features, smri_dim = load_smri_features(dataset_config, num_subjects)
#     else:
#         smri_features = np.zeros((num_subjects, 1), dtype=np.float64)
#         smri_dim = 1
        
#     # final_fc, final_pearson, labels, pseudo_arr = [torch.from_numpy(
#     #     data).float() for data in (final_fc, final_pearson, labels, pseudo_arr)]
#     final_fc, final_pearson, labels, pseudo_arr, smri_features = [
#         torch.from_numpy(data).float()
#         for data in (final_fc, final_pearson, labels, pseudo_arr, smri_features)
#     ]
#     length = final_fc.shape[0]
#     train_length = int(length*dataset_config["train_set"])
#     val_length = int(length*dataset_config["val_set"])


#     dataset = utils.TensorDataset(
#         final_fc,
#         final_pearson,
#         labels,
#         pseudo_arr,
#         smri_features
#     )

#     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#         dataset, [train_length, val_length, length-train_length-val_length])

#     train_dataloader = utils.DataLoader(
#         train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

#     val_dataloader = utils.DataLoader(
#         val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

#     test_dataloader = utils.DataLoader(
#         test_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)


#     return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries, smri_dim


def init_dataloader(dataset_config):
    data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
    final_fc = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]

    _, _, timeseries = final_fc.shape
    _, node_size, node_feature_size = final_pearson.shape

    scaler = StandardScaler(mean=np.mean(final_fc), std=np.std(final_fc))
    final_fc = scaler.transform(final_fc)

    pseudo = []
    for i in range(len(final_fc)):
        pseudo.append(np.diag(np.ones(final_pearson.shape[1])))

    if 'cc200' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 200, 200))
    elif 'aal' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 116, 116))
    elif 'cc400' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 392, 392))
    else:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))

    use_smri = dataset_config.get("use_smri", False)
    smri_encoder_type = dataset_config.get("smri_encoder_type", "fcn")
    num_subjects = final_fc.shape[0]

    view_node_features = None
    mvgcn_view_meta = None
    mvgcn_fold_graphs = None

    if use_smri and smri_encoder_type == "multiview_gcn":
        from imports.smri_graph_build import build_view_node_features, VIEW_NAMES
        _, view_node_features = build_view_node_features(dataset_config, num_subjects)

        n_nodes_per_view = {v: arr.shape[1] for v, arr in view_node_features.items()}
        n_subfeat_per_view = {v: arr.shape[2] for v, arr in view_node_features.items()}

        flat_per_view = [view_node_features[v].reshape(num_subjects, -1) for v in VIEW_NAMES]
        smri_features = np.concatenate(flat_per_view, axis=1)
        smri_dim = smri_features.shape[1]

        mvgcn_view_meta = {
            "view_names": VIEW_NAMES,
            "n_nodes_per_view": n_nodes_per_view,
            "n_subfeat_per_view": n_subfeat_per_view,
        }
    elif use_smri:
        smri_features, smri_dim = load_smri_features(dataset_config, num_subjects)
    else:
        smri_features = np.zeros((num_subjects, 1), dtype=np.float64)
        smri_dim = 1

    final_fc, final_pearson, labels, pseudo_arr, smri_features = [
        torch.from_numpy(data).float()
        for data in (final_fc, final_pearson, labels, pseudo_arr, smri_features)
    ]
    length = final_fc.shape[0]
    train_length = int(length * dataset_config["train_set"])
    val_length = int(length * dataset_config["val_set"])

    dataset = utils.TensorDataset(
        final_fc, final_pearson, labels, pseudo_arr, smri_features
    )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length, length - train_length - val_length])

    if use_smri and smri_encoder_type == "multiview_gcn":
        from imports.smri_graph_build import build_fold_graphs
        train_idx = train_dataset.indices  # torch's Subset exposes this directly
        k_per_view = dataset_config.get("mvgcn_k_neighbors", {"aseg": 8, "aparc": 32, "wmparc": 16})
        base_edge_index, base_edge_weight = build_fold_graphs(view_node_features, train_idx, k_per_view)
        mvgcn_fold_graphs = {
            "base_edge_index": base_edge_index,
            "base_edge_weight": base_edge_weight,
        }

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries, smri_dim, \
        mvgcn_view_meta, mvgcn_fold_graphs