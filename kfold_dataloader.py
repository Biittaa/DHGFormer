import numpy as np
import torch
import torch.utils.data as utils
import csv
import re

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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


def _load_raw_tensors(dataset_config):
    """Loads and preprocesses everything EXCEPT splitting into train/val/test.
    Shared by both the old single-split loader and the new k-fold loader."""
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
    num_subjects = final_fc.shape[0]
    if use_smri:
        smri_features, smri_dim = load_smri_features(dataset_config, num_subjects)
    else:
        smri_features = np.zeros((num_subjects, 1), dtype=np.float64)
        smri_dim = 1

    final_fc, final_pearson, labels_t, pseudo_arr, smri_features = [
        torch.from_numpy(d).float()
        for d in (final_fc, final_pearson, labels, pseudo_arr, smri_features)
    ]

    return {
        "final_fc": final_fc,
        "final_pearson": final_pearson,
        "labels": labels_t,
        "labels_np": labels,          # raw numpy labels, needed for StratifiedKFold
        "pseudo_arr": pseudo_arr,
        "smri_features": smri_features,
        "node_size": node_size,
        "node_feature_size": node_feature_size,
        "timeseries": timeseries,
        "smri_dim": smri_dim,
    }


def init_dataloader(dataset_config):
    """Original single random split (train_set/val_set ratios). Kept for
    backward compatibility if you ever want a quick non-CV run."""
    raw = _load_raw_tensors(dataset_config)

    dataset = utils.TensorDataset(
        raw["final_fc"], raw["final_pearson"], raw["labels"],
        raw["pseudo_arr"], raw["smri_features"]
    )

    length = raw["final_fc"].shape[0]
    train_length = int(length * dataset_config["train_set"])
    val_length = int(length * dataset_config["val_set"])

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length, length - train_length - val_length])

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), \
        raw["node_size"], raw["node_feature_size"], raw["timeseries"], raw["smri_dim"]


# def get_kfold_split_indices(labels_np, kfold, fold_idx, val_ratio=0.1, seed=123):
#     """Real stratified k-fold: (kfold-1)/kfold of the data is train+val,
#     1/kfold is held out as test for this fold. train+val is further split
#     into train/val by val_ratio, with a fixed RNG so it's reproducible
#     per fold."""
#     labels_flat = np.asarray(labels_np).reshape(-1)

#     skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
#     splits = list(skf.split(np.zeros(len(labels_flat)), labels_flat))

#     if not (0 <= fold_idx < kfold):
#         raise ValueError(f"fold_idx must be in [0, {kfold}), got {fold_idx}")

#     train_val_idx, test_idx = splits[fold_idx]

#     rng = np.random.RandomState(seed + fold_idx)
#     shuffled = train_val_idx.copy()
#     rng.shuffle(shuffled)

#     val_size = max(1, int(round(len(shuffled) * val_ratio)))
#     val_idx = shuffled[:val_size]
#     train_idx = shuffled[val_size:]

#     return train_idx, val_idx, test_idx


def get_kfold_split_indices(labels_np, kfold, fold_idx,
                            val_ratio=0.1, seed=123):
    """
    Fully stratified K-Fold split:

    Stage 1:
        Stratified K-Fold
        → Test = 1/kfold of total data

    Stage 2:
        Stratified split of remaining data
        → Validation = val_ratio of Train+Validation
        → Train = remaining samples
    """

    labels_flat = np.asarray(labels_np).reshape(-1)

    if not (0 <= fold_idx < kfold):
        raise ValueError(
            f"fold_idx must be in [0, {kfold}), got {fold_idx}"
        )

    # ==========================================
    # Stage 1: Stratified K-Fold for Test
    # ==========================================
    outer_skf = StratifiedKFold(
        n_splits=kfold,
        shuffle=True,
        random_state=seed
    )

    splits = list(
        outer_skf.split(
            np.zeros(len(labels_flat)),
            labels_flat
        )
    )

    train_val_idx, test_idx = splits[fold_idx]

    # ==========================================
    # Stage 2: Stratified split for Validation
    # ==========================================
    train_val_labels = labels_flat[train_val_idx]

    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed + fold_idx
    )

    train_relative_idx, val_relative_idx = next(
        inner_splitter.split(
            np.zeros(len(train_val_idx)),
            train_val_labels
        )
    )

    train_idx = train_val_idx[train_relative_idx]
    val_idx = train_val_idx[val_relative_idx]

    return train_idx, val_idx, test_idx

def init_dataloader_kfold(dataset_config, fold_idx, kfold=5, val_ratio=0.1, seed=123):
    """Real stratified k-fold cross validation loader.
    Call this once per fold_idx (0..kfold-1) from main.py."""
    raw = _load_raw_tensors(dataset_config)

    dataset = utils.TensorDataset(
        raw["final_fc"], raw["final_pearson"], raw["labels"],
        raw["pseudo_arr"], raw["smri_features"]
    )

    train_idx, val_idx, test_idx = get_kfold_split_indices(
        raw["labels_np"], kfold=kfold, fold_idx=fold_idx,
        val_ratio=val_ratio, seed=seed
    )

    print(f"[Fold {fold_idx+1}/{kfold}] train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)}")

    train_dataset = utils.Subset(dataset, train_idx)
    val_dataset = utils.Subset(dataset, val_idx)
    test_dataset = utils.Subset(dataset, test_idx)

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), \
        raw["node_size"], raw["node_feature_size"], raw["timeseries"], raw["smri_dim"]