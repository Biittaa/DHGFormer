# import re
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# ASEG_STYLE_SUFFIXES = ['NVoxels', 'Volume_mm3', 'normMax', 'normMean', 'normMin', 'normRange', 'normStdDev']
# APARC_STYLE_SUFFIXES = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']

# VIEW_CONFIGS = {
#     'aseg':   {'prefixes': ['aseg'],     'suffixes': ASEG_STYLE_SUFFIXES},
#     'aparc':  {'prefixes': ['lh', 'rh'], 'suffixes': APARC_STYLE_SUFFIXES},
#     'wmparc': {'prefixes': ['wmparc'],   'suffixes': ASEG_STYLE_SUFFIXES},
# }
# VIEW_NAMES = list(VIEW_CONFIGS.keys())


# def _parse_roi_columns(prefix, columns, suffixes):
#     sorted_suffixes = sorted(suffixes, key=len, reverse=True)
#     roi_map = {}
#     for col in columns:
#         if not col.startswith(prefix + '_'):
#             continue
#         remainder = col[len(prefix) + 1:]
#         for suf in sorted_suffixes:
#             if remainder.endswith('_' + suf):
#                 roi = remainder[: -(len(suf) + 1)]
#                 roi_map.setdefault(roi, {})[suf] = col
#                 break
#     return roi_map


# def _load_subject_order(order_path):
#     """Same alignment logic as dataloader.load_smri_features / kfold_dataloader's
#     version -- kept identical on purpose so fMRI/sMRI subject alignment never
#     diverges between encoders."""
#     order_df = pd.read_csv(order_path, sep="\t", header=None,
#                             names=["index_in_drive", "subject_id", "site"], skiprows=2)
#     order_df = order_df.dropna(subset=["subject_id"]).copy()
#     order_df["subject_id"] = (
#         order_df["subject_id"].astype(str).str.strip().apply(lambda x: str(int(x)))
#     )
#     order_df = order_df[order_df["subject_id"].str.fullmatch(r"\d+")].copy()
#     return order_df["subject_id"].tolist()


# def build_view_node_features(dataset_config, num_subjects):
#     """Builds, per view, a (num_subjects, n_nodes, n_subfeat) array aligned to
#     the same subject_order used by fMRI (time_series_subjects_order), z-scored
#     per column.

#     Returns:
#         view_node_names    : view -> list[str]  (ROI/node names, for reference)
#         view_node_features : view -> np.ndarray (num_subjects, n_nodes, n_subfeat)
#     """
#     subject_order = _load_subject_order(dataset_config["time_series_subjects_order"])
#     if len(subject_order) != num_subjects:
#         raise ValueError(
#             f"Subject-order file contains {len(subject_order)} subjects, "
#             f"but fMRI contains {num_subjects} subjects."
#         )

#     smri_df = pd.read_csv(dataset_config["smri_path"])
#     smri_df["SUB_ID"] = smri_df["subject_id"].apply(
#         lambda s: str(int(re.findall(r"\d+", str(s))[-1]))
#         if re.findall(r"\d+", str(s)) else None
#     )
#     smri_df = smri_df.set_index("SUB_ID")
#     # Reindex to subject_order: subjects missing from smri_df become an
#     # all-NaN row, which the "all-NaN column -> column mean -> 0" handling
#     # below already deals with (same rule as dataloader.load_smri_features).
#     smri_df = smri_df.reindex(subject_order)

#     view_node_names = {}
#     view_node_features = {}

#     for view, cfg in VIEW_CONFIGS.items():
#         roi_entries = []
#         multi_prefix = len(cfg['prefixes']) > 1
#         for prefix in cfg['prefixes']:
#             roi_map = _parse_roi_columns(prefix, smri_df.columns, cfg['suffixes'])
#             for roi_name, suf_to_col in roi_map.items():
#                 node_name = f'{prefix}_{roi_name}' if multi_prefix else roi_name
#                 col_list = [suf_to_col.get(suf) for suf in cfg['suffixes']]
#                 roi_entries.append((node_name, col_list))

#         n_nodes = len(roi_entries)
#         n_subfeat = len(cfg['suffixes'])
#         mat = np.full((num_subjects, n_nodes, n_subfeat), np.nan, dtype=np.float64)

#         for node_idx, (node_name, col_list) in enumerate(roi_entries):
#             for suf_idx, col_name in enumerate(col_list):
#                 if col_name is None:
#                     continue
#                 mat[:, node_idx, suf_idx] = pd.to_numeric(smri_df[col_name], errors='coerce').values

#         flat = mat.reshape(num_subjects, -1)
#         col_means = np.nanmean(flat, axis=0)
#         col_means = np.nan_to_num(col_means, nan=0.0)
#         nan_rows, nan_cols = np.where(np.isnan(flat))
#         flat[nan_rows, nan_cols] = col_means[nan_cols]

#         flat = StandardScaler().fit_transform(flat)
#         mat = flat.reshape(num_subjects, n_nodes, n_subfeat)

#         view_node_names[view] = [name for name, _ in roi_entries]
#         view_node_features[view] = mat.astype(np.float32)
#         print(f'[smri_graph_build] view "{view}": {n_nodes} ROI nodes x {n_subfeat} sub-features')

#     return view_node_names, view_node_features


# def _compute_pearson_similarity(node_feats):
#     sim = np.corrcoef(node_feats.T)
#     return np.nan_to_num(sim, nan=0.0)


# def build_covariance_graph(node_feats_3d_train, k):
#     """node_feats_3d_train: (n_train_subjects, n_nodes, n_subfeat) -- TRAIN
#     subjects of the current fold ONLY, so val/test never leak into graph
#     topology or weights."""
#     n_train, n_nodes, n_subfeat = node_feats_3d_train.shape
#     node_feats = node_feats_3d_train.transpose(0, 2, 1).reshape(n_train * n_subfeat, n_nodes)
#     sim = _compute_pearson_similarity(node_feats)
#     np.fill_diagonal(sim, -np.inf)

#     edge_list, weight_list = [], []
#     seen_edges = set()
#     for i in range(n_nodes):
#         neighbors = np.argsort(sim[i])[-k:]
#         for j in neighbors:
#             j = int(j)
#             if (i, j) not in seen_edges:
#                 seen_edges.add((i, j))
#                 edge_list.append([i, j]); weight_list.append(sim[i, j])
#             if (j, i) not in seen_edges:
#                 seen_edges.add((j, i))
#                 edge_list.append([j, i]); weight_list.append(sim[i, j])

#     edge_index = np.array(edge_list, dtype=np.int64).T
#     edge_weight = np.array(weight_list, dtype=np.float32)
#     return edge_index, edge_weight


# def build_fold_graphs(view_node_features, train_idx, k_per_view):
#     """Per view, builds the untiled (single-copy) kNN graph from train_idx
#     subjects only. Call once per fold, with that fold's train_idx."""
#     base_edge_index = {}
#     base_edge_weight = {}
#     for view, feats in view_node_features.items():
#         k = k_per_view.get(view, 32)
#         edge_index, edge_weight = build_covariance_graph(feats[train_idx], k=k)
#         base_edge_index[view] = edge_index
#         base_edge_weight[view] = edge_weight
#         print(f'[smri_graph_build] view "{view}": k={k}, {edge_index.shape[1]} directed edges (from train subjects only)')
#     return base_edge_index, base_edge_weight



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
    """دقیقاً معادل feature_selection() توی نوت‌بوک: RidgeClassifier + RFE,
    fit روی کل subject ها (نه فقط train) -- چون نوت‌بوک هم
    all_indices = np.arange(n_subjects) استفاده می‌کند."""
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


# def build_view_node_features(dataset_config, num_subjects):
#     """Builds, per view, a (num_subjects, n_nodes, n_subfeat) array aligned to
#     the same subject_order used by fMRI (time_series_subjects_order), z-scored
#     per column.

#     Returns:
#         view_node_names    : view -> list[str]  (ROI/node names, for reference)
#         view_node_features : view -> np.ndarray (num_subjects, n_nodes, n_subfeat)
#     """
#     subject_order = _load_subject_order(dataset_config["time_series_subjects_order"])
#     if len(subject_order) != num_subjects:
#         raise ValueError(
#             f"Subject-order file contains {len(subject_order)} subjects, "
#             f"but fMRI contains {num_subjects} subjects."
#         )

#     smri_df = pd.read_csv(dataset_config["smri_path"])
#     smri_df["SUB_ID"] = smri_df["subject_id"].apply(
#         lambda s: str(int(re.findall(r"\d+", str(s))[-1]))
#         if re.findall(r"\d+", str(s)) else None
#     )
#     smri_df = smri_df.set_index("SUB_ID")
#     # Reindex to subject_order: subjects missing from smri_df become an
#     # all-NaN row, which the "all-NaN column -> column mean -> 0" handling
#     # below already deals with (same rule as dataloader.load_smri_features).
#     smri_df = smri_df.reindex(subject_order)

#     view_node_names = {}
#     view_node_features = {}

#     for view, cfg in VIEW_CONFIGS.items():
#         roi_entries = []
#         multi_prefix = len(cfg['prefixes']) > 1
#         for prefix in cfg['prefixes']:
#             roi_map = _parse_roi_columns(prefix, smri_df.columns, cfg['suffixes'])
#             for roi_name, suf_to_col in roi_map.items():
#                 node_name = f'{prefix}_{roi_name}' if multi_prefix else roi_name
#                 col_list = [suf_to_col.get(suf) for suf in cfg['suffixes']]
#                 roi_entries.append((node_name, col_list))

#         n_nodes = len(roi_entries)
#         n_subfeat = len(cfg['suffixes'])
#         mat = np.full((num_subjects, n_nodes, n_subfeat), np.nan, dtype=np.float64)

#         for node_idx, (node_name, col_list) in enumerate(roi_entries):
#             for suf_idx, col_name in enumerate(col_list):
#                 if col_name is None:
#                     continue
#                 mat[:, node_idx, suf_idx] = pd.to_numeric(smri_df[col_name], errors='coerce').values

#         # flat = mat.reshape(num_subjects, -1)
#         # col_means = np.nanmean(flat, axis=0)
#         # col_means = np.nan_to_num(col_means, nan=0.0)
#         # nan_rows, nan_cols = np.where(np.isnan(flat))
#         # flat[nan_rows, nan_cols] = col_means[nan_cols]
#         # strategy = dataset_config.get("smri_impute_strategy", "mean")
#         # knn_k = dataset_config.get("smri_impute_knn_neighbors", 10)
#         # imputer = make_imputer(strategy, knn_k)
#         # smri_features = imputer.fit_transform(smri_features)
#         flat = mat.reshape(num_subjects, -1)

#         strategy = dataset_config.get("smri_impute_strategy", "mean")
#         knn_k = dataset_config.get("smri_impute_knn_neighbors", 10)

#         imputer = make_imputer(strategy, knn_k)
#         flat = imputer.fit_transform(flat)

#         flat = StandardScaler().fit_transform(flat)

#         mat = flat.reshape(num_subjects, n_nodes, n_subfeat)

#         flat = StandardScaler().fit_transform(flat)
#         mat = flat.reshape(num_subjects, n_nodes, n_subfeat)

#         view_node_names[view] = [name for name, _ in roi_entries]
#         view_node_features[view] = mat.astype(np.float32)
#         print(f'[smri_graph_build] view "{view}": {n_nodes} ROI nodes x {n_subfeat} sub-features')

#     return view_node_names, view_node_features












def build_view_node_features(dataset_config, num_subjects, labels=None):
    """Builds, per view, a (num_subjects, n_nodes, n_subfeat) array aligned to
    the same subject_order used by fMRI (time_series_subjects_order), z-scored
    per column.

    اگر dataset_config['use_smri_ridge_fs'] برابر True باشد، ابتدا Ridge/RFE
    (دقیقاً مثل نوت‌بوک) روی کل ماتریس sMRI اجرا می‌شود؛ هر ستون sub-feature
    که انتخاب نشود برای همه‌ی node ها NaN می‌ماند و توسط قانون قبلی
    all-NaN-column صفر می‌شود. در این حالت `labels` الزامی است.

    Returns:
        view_node_names    : view -> list[str]
        view_node_features : view -> np.ndarray (num_subjects, n_nodes, n_subfeat)
    """
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

    use_ridge_fs = dataset_config.get("use_smri_ridge_fs", False)
    selected_cols = None
    if use_ridge_fs:
        if labels is None:
            raise ValueError("build_view_node_features: labels is required when use_smri_ridge_fs=True")
        n_select = dataset_config.get("smri_ridge_num_features", 500)
        selected_cols = ridge_rfe_select_columns(smri_df, labels, n_select)

    view_node_names = {}
    view_node_features = {}

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
            print(f'[smri_graph_build] view "{view}": {n_dropped_by_ridge} sub-feature column(s) zeroed out (not selected by Ridge/RFE)')

        # flat = mat.reshape(num_subjects, -1)
        # col_means = np.nanmean(flat, axis=0)
        # col_means = np.nan_to_num(col_means, nan=0.0)
        # nan_rows, nan_cols = np.where(np.isnan(flat))
        # flat[nan_rows, nan_cols] = col_means[nan_cols]
        # flat = mat.reshape(num_subjects, -1)

        # strategy = dataset_config.get("smri_impute_strategy", "mean")
        # knn_k = dataset_config.get("smri_impute_knn_neighbors", 10)

        # imputer = make_imputer(strategy, knn_k)
        # flat = imputer.fit_transform(flat)

        # flat = StandardScaler().fit_transform(flat)

        # mat = flat.reshape(num_subjects, n_nodes, n_subfeat)

        # flat = StandardScaler().fit_transform(flat)
        # mat = flat.reshape(num_subjects, n_nodes, n_subfeat)
        flat = mat.reshape(num_subjects, -1)

        all_nan_cols = np.all(np.isnan(flat), axis=0)
        if all_nan_cols.any():
            flat[:, all_nan_cols] = 0.0

        remaining_nan = np.isnan(flat)
        if remaining_nan.any():
            strategy = dataset_config.get("smri_impute_strategy", "mean")
            knn_k = dataset_config.get("smri_impute_knn_neighbors", 10)
            imputer = make_imputer(strategy, knn_k)
            partial_cols = ~all_nan_cols
            flat[:, partial_cols] = imputer.fit_transform(flat[:, partial_cols])

        flat = StandardScaler().fit_transform(flat)
        mat = flat.reshape(num_subjects, n_nodes, n_subfeat)

        # flat = StandardScaler().fit_transform(flat)
        # mat = flat.reshape(num_subjects, n_nodes, n_subfeat)

        view_node_names[view] = [name for name, _ in roi_entries]
        view_node_features[view] = mat.astype(np.float32)
        print(f'[smri_graph_build] view "{view}": {n_nodes} ROI nodes x {n_subfeat} sub-features')

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
