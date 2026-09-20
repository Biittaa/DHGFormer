import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE


ASEG_STYLE_SUFFIXES = ['NVoxels', 'Volume_mm3', 'normMax', 'normMean', 'normMin', 'normRange', 'normStdDev']
APARC_STYLE_SUFFIXES = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']

VIEW_CONFIGS = {
    'aseg':   {'prefixes': ['aseg'],     'suffixes': ASEG_STYLE_SUFFIXES},
    'aparc':  {'prefixes': ['lh', 'rh'], 'suffixes': APARC_STYLE_SUFFIXES},
    'wmparc': {'prefixes': ['wmparc'],   'suffixes': ASEG_STYLE_SUFFIXES},
}
VIEW_NAMES = list(VIEW_CONFIGS.keys())

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def ridge_rfe_select_columns(smri_df, labels, n_select, step=100, verbose=1):
    feature_cols = [c for c in smri_df.columns
                    if c not in ("subject_id", "SUB_ID") and pd.api.types.is_numeric_dtype(smri_df[c])]

    flat = smri_df[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    col_means = np.nanmean(flat, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_rows, nan_cols = np.where(np.isnan(flat))
    flat[nan_rows, nan_cols] = col_means[nan_cols]
    flat = StandardScaler().fit_transform(flat)

    labels_flat = np.asarray(labels).reshape(-1)
    n_select = min(n_select, flat.shape[1])

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=n_select, step=step, verbose=verbose)
    selector.fit(flat, labels_flat)

    selected_cols = {feature_cols[i] for i in np.where(selector.support_)[0]}
    print(f'[smri_graph_build] Ridge RFE selected {len(selected_cols)} feature(s) out of {len(feature_cols)}')
    return selected_cols






def make_imputer(strategy='mean', knn_neighbors=10):
    if strategy == 'mean':
        return SimpleImputer(strategy='mean')
    elif strategy == 'median':
        return SimpleImputer(strategy='median')
    elif strategy == 'knn':
        return KNNImputer(n_neighbors=knn_neighbors, weights='distance')
    elif strategy == 'iterative':
        return IterativeImputer(max_iter=10, random_state=0)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

        
def _parse_roi_columns(prefix, columns, suffixes):
    sorted_suffixes = sorted(suffixes, key=len, reverse=True)
    roi_map = {}
    for col in columns:
        if not col.startswith(prefix + '_'):
            continue
        remainder = col[len(prefix) + 1:]
        for suf in sorted_suffixes:
            if remainder.endswith('_' + suf):
                roi = remainder[: -(len(suf) + 1)]
                roi_map.setdefault(roi, {})[suf] = col
                break
    return roi_map


def _load_subject_order(order_path):
    """Same alignment logic as dataloader.load_smri_features / kfold_dataloader's
    version -- kept identical on purpose so fMRI/sMRI subject alignment never
    diverges between encoders."""
    order_df = pd.read_csv(order_path, sep="\t", header=None,
                            names=["index_in_drive", "subject_id", "site"], skiprows=2)
    order_df = order_df.dropna(subset=["subject_id"]).copy()
    order_df["subject_id"] = (
        order_df["subject_id"].astype(str).str.strip().apply(lambda x: str(int(x)))
    )
    order_df = order_df[order_df["subject_id"].str.fullmatch(r"\d+")].copy()
    return order_df["subject_id"].tolist()

ICV_CANDIDATES = ['aseg_Measure_ICV', 'wmparc_Measure_ICV']
VOLUME_SUFFIXES = ('Volume_mm3', 'NVoxels', 'GrayVol')
AREA_SUFFIXES = ('SurfArea',)


def normalize_by_icv(smri_df, normalize_area=True):
    icv_col = next((c for c in ICV_CANDIDATES if c in smri_df.columns), None)
    if icv_col is None:
        raise KeyError(f"No ICV column found. Looked for {ICV_CANDIDATES}")

    df = smri_df.copy()
    icv = pd.to_numeric(df[icv_col], errors='coerce')
    icv = icv.where(icv > 0)              # صفر/منفی -> NaN (بعداً impute می‌شود)

    n_vol = n_area = 0
    for col in df.columns:
        name = str(col)
        if name.endswith(tuple('_' + s for s in VOLUME_SUFFIXES)):
            df[col] = pd.to_numeric(df[col], errors='coerce') / icv * 1000.0
            n_vol += 1
        elif normalize_area and name.endswith(tuple('_' + s for s in AREA_SUFFIXES)):
            df[col] = pd.to_numeric(df[col], errors='coerce') / (icv ** (2 / 3))
            n_area += 1

    print(f"[smri_graph_build] ICV norm using '{icv_col}': "
          f"{n_vol} volume col(s), {n_area} area col(s), "
          f"{int(icv.isna().sum())} subject(s) without valid ICV")
    return df    


def build_view_node_features(dataset_config, num_subjects, labels=None, train_idx=None, site=None):
    fit_rows = np.arange(num_subjects) if train_idx is None else np.asarray(train_idx)
    subject_order = _load_subject_order(dataset_config["time_series_subjects_order"])
    if len(subject_order) != num_subjects:
        raise ValueError(
            f"Subject-order file contains {len(subject_order)} subjects, "
            f"but fMRI contains {num_subjects} subjects."
        )

    smri_df = pd.read_csv(dataset_config["smri_path"])
    smri_df["SUB_ID"] = smri_df["subject_id"].apply(
        lambda s: str(int(re.findall(r"\d+", str(s))[-1]))
        if re.findall(r"\d+", str(s)) else None
    )
    smri_df = smri_df.set_index("SUB_ID")
    smri_df = smri_df.reindex(subject_order)
    if dataset_config.get("smri_etiv_norm", False):
            smri_df = normalize_by_icv(smri_df)

    use_ridge_fs = dataset_config.get("use_smri_ridge_fs", False)
    selected_cols = None
    if use_ridge_fs:
        if labels is None:
            raise ValueError("labels is required when use_smri_ridge_fs=True")
        n_select = dataset_config.get("smri_ridge_num_features", 500)
        selected_cols = ridge_rfe_select_columns(smri_df, labels, n_select)

    # ---- فاز ۱: ساخت raw matrix هر view، بدون ایمپیوت ----
    raw_mats = {}
    roi_entries_per_view = {}
    for view, cfg in VIEW_CONFIGS.items():
        roi_entries = []
        multi_prefix = len(cfg['prefixes']) > 1
        for prefix in cfg['prefixes']:
            roi_map = _parse_roi_columns(prefix, smri_df.columns, cfg['suffixes'])
            for roi_name, suf_to_col in roi_map.items():
                node_name = f'{prefix}_{roi_name}' if multi_prefix else roi_name
                col_list = [suf_to_col.get(suf) for suf in cfg['suffixes']]
                roi_entries.append((node_name, col_list))

        n_nodes = len(roi_entries)
        n_subfeat = len(cfg['suffixes'])
        mat = np.full((num_subjects, n_nodes, n_subfeat), np.nan, dtype=np.float64)

        n_dropped_by_ridge = 0
        for node_idx, (node_name, col_list) in enumerate(roi_entries):
            for suf_idx, col_name in enumerate(col_list):
                if col_name is None:
                    continue
                if selected_cols is not None and col_name not in selected_cols:
                    n_dropped_by_ridge += 1
                    continue
                mat[:, node_idx, suf_idx] = pd.to_numeric(smri_df[col_name], errors='coerce').values

        if selected_cols is not None and n_dropped_by_ridge:
            print(f'[smri_graph_build] view "{view}": {n_dropped_by_ridge} column(s) zeroed (not in Ridge/RFE)')

        raw_mats[view] = mat
        roi_entries_per_view[view] = roi_entries
        print(f'[smri_graph_build] view "{view}": {n_nodes} ROI nodes x {n_subfeat} sub-features')

    flats = [raw_mats[v].reshape(num_subjects, -1) for v in VIEW_NAMES]
    split_sizes = [f.shape[1] for f in flats]
    master_flat = np.concatenate(flats, axis=1)

    all_nan_cols = np.all(np.isnan(master_flat), axis=0)
    if all_nan_cols.any():
        master_flat[:, all_nan_cols] = 0.0

    remaining_nan = np.isnan(master_flat)
    if remaining_nan.any():
        strategy = dataset_config.get("smri_impute_strategy", "mean")
        knn_k = dataset_config.get("smri_impute_knn_neighbors", 10)
        imputer = make_imputer(strategy, knn_k)
        partial_cols = ~all_nan_cols
        master_flat[:, partial_cols] = imputer.fit_transform(master_flat[:, partial_cols])
    
        if dataset_config.get("use_combat", False):
            from neuroHarmonize import harmonizationLearn, harmonizationApply

        pheno = pd.read_csv(dataset_config["pheno_path"])
        pheno["SUB_ID"] = pheno["SUB_ID"].astype(int).astype(str)
        pheno = pheno.set_index("SUB_ID").reindex(subject_order)
        if pheno["SITE_ID"].isna().any():
            raise ValueError("Some subjects in subject_order are missing from the phenotypic file.")

        age = pd.to_numeric(pheno["AGE_AT_SCAN"], errors="coerce")
        age = age.where(age > 0)
        age = age.fillna(age.iloc[fit_rows].median())      # median فقط از train
        sex = (pd.to_numeric(pheno["SEX"], errors="coerce") == 2).astype(int)

        covars = pd.DataFrame({
            "SITE": pheno["SITE_ID"].astype(str).values,
            "AGE": age.values,
            "SEX": sex.values,
        })

        if site is not None:
            n_bad = int((np.asarray(site).astype(str) != covars["SITE"].values).sum())
            print(f"[smri_graph_build] site mismatch vs abide.npy: {n_bad}")

        train_sites = set(covars["SITE"].iloc[fit_rows])
        unseen = set(covars["SITE"]) - train_sites
        if unseen:
            raise ValueError(f"Sites with no train subject in this split: {unseen}")

        tr = master_flat[fit_rows]
        site_tr = covars["SITE"].iloc[fit_rows].values
        ok = tr.std(axis=0) > 1e-8
        for s in train_sites:
            ok &= tr[site_tr == s].std(axis=0) > 1e-8

        model, _ = harmonizationLearn(
            tr[:, ok], covars.iloc[fit_rows].reset_index(drop=True))
        master_flat[:, ok] = harmonizationApply(
            master_flat[:, ok], covars.reset_index(drop=True), model)

        if not np.isfinite(master_flat).all():
            raise ValueError("Non-finite values after ComBat")
        print(f"[smri_graph_build] ComBat (SITE+AGE+SEX): harmonized {int(ok.sum())}/{ok.size} columns")
    # if dataset_config.get("use_combat", False):
    #     from neuroHarmonize import harmonizationLearn, harmonizationApply
    #     covars = pd.DataFrame({"SITE": np.asarray(site).astype(str)})
    #     ok = master_flat[fit_rows].std(axis=0) > 1e-8      # ستون ثابت در train را کنار بگذار
    #     model, _ = harmonizationLearn(master_flat[fit_rows][:, ok],
    #                                   covars.iloc[fit_rows].reset_index(drop=True))
    #     master_flat[:, ok] = harmonizationApply(master_flat[:, ok],
    #                                             covars.reset_index(drop=True), model)

    
    
    
    view_node_names = {}
    view_node_features = {}
    offset = 0
    for view, size in zip(VIEW_NAMES, split_sizes):
        n_nodes = raw_mats[view].shape[1]
        n_subfeat = raw_mats[view].shape[2]
        flat_v = master_flat[:, offset:offset + size]
        offset += size

        flat_v = StandardScaler().fit_transform(flat_v)
        mat = flat_v.reshape(num_subjects, n_nodes, n_subfeat)

        view_node_names[view] = [name for name, _ in roi_entries_per_view[view]]
        view_node_features[view] = mat.astype(np.float32)

    return view_node_names, view_node_features

























def _compute_pearson_similarity(node_feats):
    sim = np.corrcoef(node_feats.T)
    return np.nan_to_num(sim, nan=0.0)


def build_covariance_graph(node_feats_3d_train, k):
    """node_feats_3d_train: (n_train_subjects, n_nodes, n_subfeat) -- TRAIN
    subjects of the current fold ONLY, so val/test never leak into graph
    topology or weights."""
    n_train, n_nodes, n_subfeat = node_feats_3d_train.shape
    node_feats = node_feats_3d_train.transpose(0, 2, 1).reshape(n_train * n_subfeat, n_nodes)
    sim = _compute_pearson_similarity(node_feats)
    np.fill_diagonal(sim, -np.inf)

    edge_list, weight_list = [], []
    seen_edges = set()
    for i in range(n_nodes):
        neighbors = np.argsort(sim[i])[-k:]
        for j in neighbors:
            j = int(j)
            if (i, j) not in seen_edges:
                seen_edges.add((i, j))
                edge_list.append([i, j]); weight_list.append(sim[i, j])
            if (j, i) not in seen_edges:
                seen_edges.add((j, i))
                edge_list.append([j, i]); weight_list.append(sim[i, j])

    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_weight = np.array(weight_list, dtype=np.float32)
    return edge_index, edge_weight


def build_fold_graphs(view_node_features, train_idx, k_per_view):
    """Per view, builds the untiled (single-copy) kNN graph from train_idx
    subjects only. Call once per fold, with that fold's train_idx."""
    base_edge_index = {}
    base_edge_weight = {}
    for view, feats in view_node_features.items():
        k = k_per_view.get(view, 32)
        edge_index, edge_weight = build_covariance_graph(feats[train_idx], k=k)
        base_edge_index[view] = edge_index
        base_edge_weight[view] = edge_weight
        print(f'[smri_graph_build] view "{view}": k={k}, {edge_index.shape[1]} directed edges (from train subjects only)')
    return base_edge_index, base_edge_weight
