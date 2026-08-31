import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ASEG_STYLE_SUFFIXES = ['NVoxels', 'Volume_mm3', 'normMax', 'normMean', 'normMin', 'normRange', 'normStdDev']
APARC_STYLE_SUFFIXES = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']

VIEW_CONFIGS = {
    'aseg':   {'prefixes': ['aseg'],     'suffixes': ASEG_STYLE_SUFFIXES},
    'aparc':  {'prefixes': ['lh', 'rh'], 'suffixes': APARC_STYLE_SUFFIXES},
    'wmparc': {'prefixes': ['wmparc'],   'suffixes': ASEG_STYLE_SUFFIXES},
}
VIEW_NAMES = list(VIEW_CONFIGS.keys())


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


def build_view_node_features(dataset_config, num_subjects):
    """Builds, per view, a (num_subjects, n_nodes, n_subfeat) array aligned to
    the same subject_order used by fMRI (time_series_subjects_order), z-scored
    per column.

    Returns:
        view_node_names    : view -> list[str]  (ROI/node names, for reference)
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
    # Reindex to subject_order: subjects missing from smri_df become an
    # all-NaN row, which the "all-NaN column -> column mean -> 0" handling
    # below already deals with (same rule as dataloader.load_smri_features).
    smri_df = smri_df.reindex(subject_order)

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

        for node_idx, (node_name, col_list) in enumerate(roi_entries):
            for suf_idx, col_name in enumerate(col_list):
                if col_name is None:
                    continue
                mat[:, node_idx, suf_idx] = pd.to_numeric(smri_df[col_name], errors='coerce').values

        flat = mat.reshape(num_subjects, -1)
        col_means = np.nanmean(flat, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        nan_rows, nan_cols = np.where(np.isnan(flat))
        flat[nan_rows, nan_cols] = col_means[nan_cols]

        flat = StandardScaler().fit_transform(flat)
        mat = flat.reshape(num_subjects, n_nodes, n_subfeat)

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
