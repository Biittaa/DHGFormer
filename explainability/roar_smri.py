#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roar_captum_dhgformer.py
=========================

ROAR (RemOve And Retrain) explainability for the **real** DHGFormer repo,
using the **real** `MultiViewGCN` sMRI encoder in `smri_only` mode, attributed
with **Captum** -- all 7 methods: Integrated Gradients, Guided Backprop,
DeepLift, DeepLiftShap, GradientShap, LIME, KernelSHAP -- run through
**main.py's** data path (`dataloader.init_dataloader`, single train/val/test
split) -- NOT the k-fold path.

This does NOT reimplement the model or the data pipeline. It imports and
calls your existing `dataloader.init_dataloader`, `model.DHGFormer.DHGFormer`
and `train.BasicTrain` completely unmodified, so the baseline ("threshold=0")
run is guaranteed to reproduce exactly what `main.py` itself would produce
for the chosen repeat, given the same config file and the same seed.

--------------------------------------------------------------------------
WHY / HOW EXACT REPRODUCIBILITY IS GUARANTEED
--------------------------------------------------------------------------
`main.py`'s repeat loop is:

    for i in range(args.repeat_time):
        current_seed = seed + i
        metrics = main(args, current_seed)

`main(args, current_seed)` fully resets every RNG (random / numpy / torch /
cuda) at the top, *before* touching the dataset or the model, and every
repeat opens a fresh copy of the yaml config. That means each repeat is
completely self-contained: the result for `current_seed = base_seed + (k-1)`
("fold k" in your terms) does not depend on anything that happened in
repeats 1..k-1. So we can reproduce "fold k" in total isolation, in a
separate process, just by calling the exact same code with the same
`current_seed` -- which is exactly what `reproduce_repeat()` below does (it
is a byte-for-byte copy of `main.py`'s `main()` function body).

For ROAR itself, at threshold > 0 we do NOT rebuild the data pipeline
differently -- we call `init_dataloader()` exactly the same way, and only
*after* it returns, we zero out specific columns of the already-built
`smri_features` tensor (shared, in-place, across train/val/test since they
are `Subset`s of the same `TensorDataset`). This preserves:
  - the exact same train/val/test split (computed before the zeroing),
  - the exact same sMRI per-view kNN graph topology/weights (also computed
    from the untouched features, before the zeroing -- graph topology is
    frozen, only node *features* are zeroed, per the ROAR paper's protocol),
  - the exact same RNG-consumption order during training (model init,
    DataLoader shuffling, etc.), since zeroing a tensor's values doesn't
    consume any RNG and doesn't change any shape.
So threshold=0.0 with an EMPTY zero-set reproduces the baseline bit-for-bit
(same seed => same result), and every other threshold differs from it ONLY
in which sMRI values were replaced with 0.

Per your instruction, threshold=0.0 does NOT retrain at all -- it just
reuses the metrics already recorded by the one baseline training run (which
is also the run whose weights get checkpointed to Drive). Every other
threshold DOES retrain from scratch, for the full configured number of
epochs (no shortcuts), with a full RNG reset first, exactly mirroring how
`main.py` itself would train that repeat.

--------------------------------------------------------------------------
WHAT GETS ATTRIBUTED / ZEROED
--------------------------------------------------------------------------
Only the `smri_only` branch of `DHGFormer.forward` is explained (matches
your yaml: `smri_only: true`, `smri_encoder_type: multiview_gcn`). Captum
attributes directly over the REAL flat `smri_features` tensor that
`dataloader.py` builds (shape: `(n_subjects, D)`, `D` = sum over views of
`n_nodes * n_subfeat`, plus any extra global/pheno columns). No reshaping
trick is needed: `DHGFormer._forward_mvgcn` already accepts an arbitrary
number of rows and infers batch size from `x.shape[0]`, so Captum's
IG/DeepLift interpolation "replica" rows just look like extra pseudo-
subjects to the model -- nothing about the real forward path is touched.

Ranking / zeroing granularity is `(view, node, sub_feature)` for the 3
per-view branches (aseg / aparc / wmparc) plus a separate `extra` group for
any `use_smri_global` / `use_pheno` columns appended after the views.

--------------------------------------------------------------------------
WHAT GETS SAVED TO DRIVE (all APPEND-ONLY -- nothing is ever overwritten
or deleted on a re-run; every run gets a unique run_id)
--------------------------------------------------------------------------
Under `--out_dir` (point this at a folder on your mounted Drive):

  baseline/
    baseline_<run_id>.pt                 state_dict of the trained fold-k model
    baseline_manifest_<run_id>.json      FULL config + seed + all metrics + env info

  importance/
    feature_importance_<run_id>_<method>.csv   full per-(view,node,subfeature) table
    importance_index.csv                       append-only index of every importance
                                                 table ever produced (run_id, method,
                                                 params, fold, path, timestamp)

  roar_results_history.csv               append-only: one row per (run_id, method,
                                          threshold) with test/val/train accuracy +
                                          auc/sen/spe/f1 + every method parameter
                                          (ig steps, deeplift baseline type, target
                                          mode, epochs used, etc.)

--------------------------------------------------------------------------
USAGE (run from the repo root, i.e. next to main.py / dataloader.py / model/)
--------------------------------------------------------------------------
    python roar_captum_dhgformer.py \
        --config_filename setting/abide_DHGFormer.yaml \
        --target_fold 4 \
        --base_seed 21 \
        --device 0 \
        --out_dir /content/drive/MyDrive/DHGFormer/roar_explainability \
        --methods ig deeplift \
        --ig_steps 64 \
        --deeplift_baseline zero \
        --target_class_mode true

All 7 Captum methods are available via `--methods`: ig, guided_backprop,
deeplift, deeplift_shap, gradient_shap, lime, kernel_shap -- plus `random`
as a sanity-comparison baseline (exactly like the paper's Fig. 5 / your own
roar_analysis.py). lime/kernel_shap are perturbation-based and run
per-subject (much slower than the gradient-based methods) -- tune
--lime_n_samples / --kernelshap_n_samples accordingly.

To only (re)compute the baseline and stop (no captum / no ROAR retraining
loop), pass `--baseline_only`.
"""

import argparse
import contextlib
import copy
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml


# --------------------------------------------------------------------------
# 0) Make sure we can import the repo's own modules (dataloader.py,
#    model/DHGFormer.py, train.py) no matter where this script is invoked
#    from. Default: the directory this script lives in.
# --------------------------------------------------------------------------
def _add_repo_root_to_path(repo_root: str):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# --------------------------------------------------------------------------
# 1) Exact reproduction of main.py's per-repeat training. This function's
#    body is intentionally a near-literal copy of main.py's `main()` -- do
#    not "clean it up", the whole point is byte-for-byte fidelity.
# --------------------------------------------------------------------------
def reproduce_repeat(config_filename: str, current_seed: int, device_index: int):
    """Runs exactly what `main.py`'s `main(args, current_seed)` runs for one
    repeat, unmodified. Returns everything the ROAR/captum code needs
    afterwards (model, dataloaders, config actually used, metrics, view
    meta) without touching main.py itself."""
    from dataloader import init_dataloader
    from model.DHGFormer import DHGFormer
    from train import BasicTrain

    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)

    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    random.seed(current_seed)
    np.random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)

    dataloaders, node_size, node_feature_size, timeseries_size, smri_dim, \
        mvgcn_view_meta, mvgcn_fold_graphs = init_dataloader(config['data'])

    config['train']["seq_len"] = timeseries_size
    config['train']["node_size"] = node_size
    config['model']['smri_encoder_type'] = config['data'].get('smri_encoder_type', 'fcn')

    model = DHGFormer(config['model'], node_size,
                       node_feature_size, timeseries_size,
                       use_smri=config['data'].get('use_smri', False),
                       smri_input_dim=smri_dim,
                       mvgcn_view_meta=mvgcn_view_meta,
                       mvgcn_fold_graphs=mvgcn_fold_graphs)

    no_decay_params, decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'temporal_encoder.gate' in name or 'temporal_encoder.pos_embed' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.Adam([
        {'params': decay_params, 'weight_decay': config['train']['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=config['train']['lr'])
    opts = (optimizer,)

    save_folder_name = Path(config['train']['log_folder']) / Path(config['model']['type']) / Path(
        f"{config['data']['dataset']}_{config['data']['atlas']}")

    train_process = BasicTrain(config['train'], model, opts, dataloaders, save_folder_name)
    metrics = train_process.train()

    return {
        "model": train_process.model,          # trained weights (see note in module docstring
                                                 # re: best_model being an alias, not a copy --
                                                 # this is exactly what train.py itself saves)
        "dataloaders": dataloaders,
        "config": config,
        "current_seed": current_seed,
        "metrics": metrics,
        "mvgcn_view_meta": mvgcn_view_meta,
    }


def zero_smri_columns_in_place(train_dataloader, col_indices):
    """Zeroes the given flat-column indices of the shared smri_features
    tensor UNDERLYING the train/val/test Subsets (all three point at the
    same TensorDataset instance built inside init_dataloader). Must be
    called AFTER init_dataloader() returns and BEFORE the model/optimizer
    are constructed or training starts -- exactly the point in the call
    sequence where reproduce_repeat() hands control back, i.e. right
    between init_dataloader() and DHGFormer(...) if you want to affect
    training, or (as this script does) by re-running reproduce_repeat with
    a hook -- see `train_with_smri_override` below."""
    if not col_indices:
        return
    base_dataset = train_dataloader.dataset.dataset  # Subset -> underlying TensorDataset
    smri_tensor = base_dataset.tensors[4]             # (final_fc, pearson, labels, pseudo, smri)
    with torch.no_grad():
        smri_tensor[:, col_indices] = 0.0


def train_with_smri_override(config_filename: str, current_seed: int, device_index: int,
                              col_indices_to_zero):
    """Same as reproduce_repeat(), but zeroes the given flat sMRI column
    indices immediately after init_dataloader() returns and before the
    model is built / trained. This keeps the RNG-consumption order and the
    train/val/test split and the sMRI graph topology IDENTICAL to the
    baseline run -- only the sMRI feature *values* differ."""
    from dataloader import init_dataloader
    from model.DHGFormer import DHGFormer
    from train import BasicTrain

    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)

    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    random.seed(current_seed)
    np.random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)

    dataloaders, node_size, node_feature_size, timeseries_size, smri_dim, \
        mvgcn_view_meta, mvgcn_fold_graphs = init_dataloader(config['data'])

    train_dataloader = dataloaders[0]
    zero_smri_columns_in_place(train_dataloader, col_indices_to_zero)

    config['train']["seq_len"] = timeseries_size
    config['train']["node_size"] = node_size
    config['model']['smri_encoder_type'] = config['data'].get('smri_encoder_type', 'fcn')

    model = DHGFormer(config['model'], node_size,
                       node_feature_size, timeseries_size,
                       use_smri=config['data'].get('use_smri', False),
                       smri_input_dim=smri_dim,
                       mvgcn_view_meta=mvgcn_view_meta,
                       mvgcn_fold_graphs=mvgcn_fold_graphs)

    no_decay_params, decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'temporal_encoder.gate' in name or 'temporal_encoder.pos_embed' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.Adam([
        {'params': decay_params, 'weight_decay': config['train']['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=config['train']['lr'])
    opts = (optimizer,)

    save_folder_name = Path(config['train']['log_folder']) / Path(config['model']['type']) / Path(
        f"{config['data']['dataset']}_{config['data']['atlas']}_roar_tmp")

    train_process = BasicTrain(config['train'], model, opts, dataloaders, save_folder_name)
    metrics = train_process.train()
    return metrics


# --------------------------------------------------------------------------
# 2) Column-offset bookkeeping: map (view, node_idx, subfeat_idx) <-> flat
#    column index in the smri_features tensor, and human-readable names for
#    the importance CSVs. Pure, RNG-free, computed AFTER the baseline run so
#    it can never perturb the reproduction above.
# --------------------------------------------------------------------------
def build_offset_table(mvgcn_view_meta):
    view_names = mvgcn_view_meta["view_names"]
    n_nodes_per_view = mvgcn_view_meta["n_nodes_per_view"]
    n_subfeat_per_view = mvgcn_view_meta["n_subfeat_per_view"]
    extra_dim = mvgcn_view_meta.get("extra_dim", 0)

    offsets = {}
    cursor = 0
    for v in view_names:
        offsets[v] = cursor
        cursor += n_nodes_per_view[v] * n_subfeat_per_view[v]
    extra_offset = cursor
    total_dim = cursor + extra_dim
    return {
        "view_names": view_names,
        "n_nodes_per_view": n_nodes_per_view,
        "n_subfeat_per_view": n_subfeat_per_view,
        "view_col_offset": offsets,
        "extra_offset": extra_offset,
        "extra_dim": extra_dim,
        "total_dim": total_dim,
    }


def col_of(offset_table, view, node_idx, subfeat_idx):
    n_subfeat = offset_table["n_subfeat_per_view"][view]
    return offset_table["view_col_offset"][view] + node_idx * n_subfeat + subfeat_idx


def get_feature_names(dataset_config, offset_table):
    """Lightweight, RNG-free: reuses the repo's own column-parsing helper
    (imports.smri_graph_build) purely to label columns for the CSVs. Never
    touches torch's RNG and is safe to call any time."""
    import pandas as pd
    from imports.smri_graph_build import VIEW_CONFIGS, _parse_roi_columns

    smri_df_cols = pd.read_csv(dataset_config["smri_path"], nrows=0).columns

    view_node_names = {}
    for view, cfg in VIEW_CONFIGS.items():
        roi_entries = []
        multi_prefix = len(cfg['prefixes']) > 1
        for prefix in cfg['prefixes']:
            roi_map = _parse_roi_columns(prefix, smri_df_cols, cfg['suffixes'])
            for roi_name in roi_map.keys():
                node_name = f'{prefix}_{roi_name}' if multi_prefix else roi_name
                roi_entries.append(node_name)
        view_node_names[view] = roi_entries

    extra_names = []
    if offset_table["extra_dim"] > 0:
        try:
            from imports.smri_graph_build import build_extra_features
            _, extra_names = build_extra_features(dataset_config, num_subjects=1, train_idx=None)
        except Exception:
            extra_names = [f"extra_{i}" for i in range(offset_table["extra_dim"])]
    return view_node_names, extra_names


# --------------------------------------------------------------------------
# 3) Captum wrapper -- reuses the REAL model's own _forward_mvgcn /
#    smri_only_classifier, no reimplementation.
# --------------------------------------------------------------------------
class SMRIOnlyCaptumWrapper(nn.Module):
    def __init__(self, dhg_model):
        super().__init__()
        assert dhg_model.use_smri and dhg_model.smri_only, \
            "Model must be built with use_smri=True and smri_only=True (see yaml)."
        assert dhg_model.smri_encoder_type == "multiview_gcn", \
            "This wrapper is for smri_encoder_type: multiview_gcn only."
        self.m = dhg_model

    def forward(self, smri_flat):
        emb = self.m._forward_mvgcn(smri_flat)
        return self.m.smri_only_classifier(emb)


@contextlib.contextmanager
def _non_inplace_relu(model):
    """Captum's DeepLift (and to a lesser extent IG) relies on module hooks
    that can misbehave with in-place ReLUs (MultiViewGCN uses
    nn.ReLU(inplace=True)). This toggles every ReLU to non-inplace only for
    the duration of attribution computation, then restores it exactly --
    weights, architecture and training are completely untouched by this."""
    toggled = []
    for mod in model.modules():
        if isinstance(mod, nn.ReLU) and mod.inplace:
            mod.inplace = False
            toggled.append(mod)
    try:
        yield
    finally:
        for mod in toggled:
            mod.inplace = True


def _select_targets(logits, labels, target_mode):
    if target_mode == "true":
        return labels.long()
    if target_mode == "predicted":
        return logits.detach().argmax(dim=1)
    if str(target_mode).lstrip("-").isdigit():
        fixed = int(target_mode)
        return torch.full((logits.shape[0],), fixed, dtype=torch.long, device=logits.device)
    raise ValueError(f"Unknown target_mode: {target_mode!r}")


GRADIENT_BASED_METHODS = {"ig", "guided_backprop", "deeplift", "deeplift_shap", "gradient_shap"}
PERTURBATION_BASED_METHODS = {"lime", "kernel_shap"}
ALL_CAPTUM_METHODS = GRADIENT_BASED_METHODS | PERTURBATION_BASED_METHODS


def build_node_feature_mask(offset_table, device):
    """Groups every flat column into its (view, node) group -- all
    sub-features of the same ROI node get the same group id -- plus one
    group per extra (global/pheno) column. This is what LIME/KernelSHAP
    perturb as a single unit; without this, the interpretable space would
    be the full flat dimension (often >1000), which is intractable for
    perturbation-based methods. Returns a (1, total_dim) LongTensor, as
    Captum's `feature_mask` expects."""
    total_dim = offset_table["total_dim"]
    mask = torch.zeros(total_dim, dtype=torch.long)
    group_id = 0
    for v in offset_table["view_names"]:
        n_nodes = offset_table["n_nodes_per_view"][v]
        n_subfeat = offset_table["n_subfeat_per_view"][v]
        start = offset_table["view_col_offset"][v]
        for node_idx in range(n_nodes):
            mask[start + node_idx * n_subfeat: start + (node_idx + 1) * n_subfeat] = group_id
            group_id += 1
    for i in range(offset_table["extra_dim"]):
        mask[offset_table["extra_offset"] + i] = group_id
        group_id += 1
    return mask.unsqueeze(0).to(device)


def gather_background_samples(train_dataloader, n_samples, mode, device, seed):
    """Background/reference set for DeepLiftShap & GradientShap.
    mode: 'zero' -> n_samples copies of the zero vector.
          'train_sample' -> n_samples subjects randomly drawn from the train
          split (deterministic given `seed`; called AFTER training is done,
          so this never perturbs the reproduction of the baseline run)."""
    all_smri = []
    for _data_in, _pearson, _label, _pseudo, smri in train_dataloader:
        all_smri.append(smri)
    all_smri = torch.cat(all_smri, dim=0)

    if mode == "zero":
        return torch.zeros(n_samples, all_smri.shape[1], device=device)
    if mode == "train_sample":
        rng = np.random.default_rng(seed)
        n_avail = all_smri.shape[0]
        idx = rng.choice(n_avail, size=min(n_samples, n_avail), replace=(n_samples > n_avail))
        return all_smri[idx].to(device).float()
    raise ValueError(f"Unsupported shap_background mode: {mode!r}")


def compute_attributions(model, train_dataloader, method, offset_table, device,
                          ig_steps=64, deeplift_baseline="zero", target_mode="true",
                          shap_background="train_sample", n_background_samples=20,
                          gradientshap_n_samples=20, gradientshap_stdev=0.0,
                          lime_n_samples=200, kernelshap_n_samples=200,
                          lime_group_by_node=True, kernelshap_group_by_node=True,
                          seed=0):
    """Runs the requested Captum method over every subject in the TRAIN
    split (never val/test, to avoid leaking split information into the
    ranking used for ROAR), and returns the mean |attribution| as a flat
    numpy vector of length offset_table['total_dim'].

    method: one of ALL_CAPTUM_METHODS
      - ig, guided_backprop, deeplift, deeplift_shap, gradient_shap:
        gradient/hook-based, run batched (fast).
      - lime, kernel_shap: perturbation-based; Captum explains one instance
        at a time for these, so they're run per-subject (slow -- tune
        lime_n_samples / kernelshap_n_samples accordingly). By default the
        interpretable space is grouped per ROI node (see
        build_node_feature_mask) rather than per raw column, or these would
        be intractable given hundreds-to-thousands of raw sMRI columns.
    """
    from captum.attr import (
        IntegratedGradients, GuidedBackprop, DeepLift, DeepLiftShap,
        GradientShap, Lime, KernelShap,
    )
    if method not in ALL_CAPTUM_METHODS:
        raise ValueError(f"Unsupported captum method: {method!r}")

    wrapper = SMRIOnlyCaptumWrapper(model).to(device)
    wrapper.eval()

    total_dim = offset_table["total_dim"]
    abs_sum = torch.zeros(total_dim, dtype=torch.float64)
    n_seen = 0

    with _non_inplace_relu(wrapper):

        # ---------------- gradient / hook-based methods (batched) ----------------
        if method in GRADIENT_BASED_METHODS:
            if method == "ig":
                explainer = IntegratedGradients(wrapper)
            elif method == "guided_backprop":
                explainer = GuidedBackprop(wrapper)
            elif method == "deeplift":
                explainer = DeepLift(wrapper)
            elif method == "deeplift_shap":
                explainer = DeepLiftShap(wrapper)
            elif method == "gradient_shap":
                explainer = GradientShap(wrapper)

            background = None
            if method in ("deeplift_shap", "gradient_shap"):
                background = gather_background_samples(
                    train_dataloader, n_background_samples, shap_background, device, seed)

            for data_in, pearson, label, _pseudo, smri in train_dataloader:
                smri = smri.to(device).float()
                label = label.to(device).long().view(-1)
                smri.requires_grad_(True)

                with torch.no_grad():
                    logits = wrapper(smri)
                targets = _select_targets(logits, label, target_mode)

                if method == "guided_backprop":
                    attr = explainer.attribute(smri, target=targets)
                elif method == "ig":
                    baselines = torch.zeros_like(smri)
                    attr = explainer.attribute(smri, baselines=baselines, target=targets,
                                                n_steps=ig_steps)
                elif method == "deeplift":
                    if deeplift_baseline == "zero":
                        baselines = torch.zeros_like(smri)
                    else:  # mean
                        baselines = smri.mean(dim=0, keepdim=True).expand_as(smri).clone()
                    attr = explainer.attribute(smri, baselines=baselines, target=targets)
                elif method == "deeplift_shap":
                    attr = explainer.attribute(smri, baselines=background, target=targets)
                elif method == "gradient_shap":
                    attr = explainer.attribute(smri, baselines=background, target=targets,
                                                n_samples=gradientshap_n_samples,
                                                stdevs=gradientshap_stdev)

                abs_sum += attr.detach().abs().double().sum(dim=0).cpu()
                n_seen += smri.shape[0]

        # ---------------- perturbation-based methods (per-subject) ----------------
        else:
            group_by_node = lime_group_by_node if method == "lime" else kernelshap_group_by_node
            feature_mask = build_node_feature_mask(offset_table, device) if group_by_node else None
            n_samples = lime_n_samples if method == "lime" else kernelshap_n_samples
            explainer = Lime(wrapper) if method == "lime" else KernelShap(wrapper)

            for data_in, pearson, label, _pseudo, smri in train_dataloader:
                smri = smri.to(device).float()
                label = label.to(device).long().view(-1)

                with torch.no_grad():
                    logits = wrapper(smri)
                targets = _select_targets(logits, label, target_mode)

                for i in range(smri.shape[0]):
                    x_i = smri[i:i + 1]
                    baseline_i = torch.zeros_like(x_i)
                    fmask_i = feature_mask if feature_mask is not None else None
                    attr_i = explainer.attribute(
                        x_i, baselines=baseline_i, target=int(targets[i].item()),
                        feature_mask=fmask_i, n_samples=n_samples)
                    abs_sum += attr_i.detach().abs().double().squeeze(0).cpu()
                    n_seen += 1

    return (abs_sum / max(n_seen, 1)).numpy()


def get_random_ranking(offset_table, seed):
    total_dim = offset_table["total_dim"]
    idx = np.arange(total_dim)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return idx.tolist()   # order only; "score" is meaningless for random


def rank_columns_by_score(score_vector):
    """score_vector: 1D np.ndarray, length total_dim. Returns column indices
    sorted by |score| descending (this is the removal order ROAR uses)."""
    order = np.argsort(-np.abs(score_vector))
    return order.tolist()


# --------------------------------------------------------------------------
# 4) ROAR loop
# --------------------------------------------------------------------------
DEFAULT_ROAR_THRESHOLDS = [0.0, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4,
                           0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def run_roar(method_name, ranking_cols, baseline_metrics, config_filename, current_seed,
             device_index, thresholds, recompute_threshold_zero=False):
    total_dim = len(ranking_cols)
    rows = []
    for frac in thresholds:
        if frac <= 0.0 and not recompute_threshold_zero:
            metrics = baseline_metrics
            retrained = False
        else:
            n_remove = int(round(frac * total_dim))
            cols_to_zero = ranking_cols[:n_remove]
            metrics = train_with_smri_override(config_filename, current_seed, device_index,
                                                 cols_to_zero)
            retrained = True
        row = {
            "method": method_name,
            "threshold": frac,
            "n_features_removed": int(round(frac * total_dim)),
            "n_features_total": total_dim,
            "retrained": retrained,
        }
        row.update({f"metric_{k}": v for k, v in metrics.items()})
        rows.append(row)
        print(f"  [{method_name}] threshold={frac:.3f} "
              f"(removed {row['n_features_removed']}/{total_dim}) "
              f"-> test_acc={metrics.get('test_acc'):.4f} "
              f"(retrained={retrained})")
    return rows


# --------------------------------------------------------------------------
# 5) Append-only logging helpers
# --------------------------------------------------------------------------
def append_csv(path, rows_df):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    rows_df.to_csv(path, mode="a", header=header, index=False)


def save_json(path, obj):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def env_info():
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name()
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except Exception:
        pass
    try:
        import captum
        info["captum"] = captum.__version__
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# 6) Orchestration
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_root", default=os.path.dirname(os.path.abspath(__file__)),
                     help="Folder containing main.py/dataloader.py/model/ (default: this script's folder)")
    ap.add_argument("--config_filename", default="setting/abide_DHGFormer.yaml")
    ap.add_argument("--target_fold", type=int, default=4,
                     help="1-indexed repeat number to reproduce, matching main.py's "
                          "`for i in range(repeat_time): current_seed = base_seed + i` "
                          "(fold k -> current_seed = base_seed + (k-1)).")
    ap.add_argument("--base_seed", type=int, default=21, help="Matches main.py --seed default.")
    ap.add_argument("--device", type=int, default=4, help="Matches main.py --device default.")
    ap.add_argument("--out_dir", required=True,
                     help="Drive folder to save baseline checkpoint, importance tables and "
                          "the append-only ROAR history CSV into.")
    ap.add_argument("--methods", nargs="+", default=["ig", "deeplift"],
                     choices=["ig", "guided_backprop", "deeplift", "deeplift_shap",
                              "gradient_shap", "lime", "kernel_shap", "random"])
    ap.add_argument("--target_class_mode", default="true",
                     help="'true' | 'predicted' | an integer class index. Applies to every "
                          "captum method (not to 'random').")
    # -- Integrated Gradients --
    ap.add_argument("--ig_steps", type=int, default=64)
    # -- DeepLift (single baseline) --
    ap.add_argument("--deeplift_baseline", choices=["zero", "mean"], default="zero")
    # -- DeepLiftShap / GradientShap (background/reference SET, not a single baseline) --
    ap.add_argument("--shap_background", choices=["zero", "train_sample"], default="train_sample",
                     help="Reference set for deeplift_shap/gradient_shap: n_background_samples "
                          "copies of zero, or that many subjects sampled from the train split.")
    ap.add_argument("--n_background_samples", type=int, default=20,
                     help="Size of the background/reference set for deeplift_shap/gradient_shap.")
    ap.add_argument("--gradientshap_n_samples", type=int, default=20,
                     help="GradientShap's internal number of randomized samples per input.")
    ap.add_argument("--gradientshap_stdev", type=float, default=0.0,
                     help="GradientShap's Gaussian noise stdev added around each baseline.")
    # -- LIME / KernelSHAP (perturbation-based, run per-subject) --
    ap.add_argument("--lime_n_samples", type=int, default=200,
                     help="Perturbed samples per subject for LIME. Runtime scales with "
                          "n_train_subjects * lime_n_samples -- tune down if too slow.")
    ap.add_argument("--kernelshap_n_samples", type=int, default=200,
                     help="Perturbed samples per subject for KernelSHAP. Same cost note as LIME.")
    ap.add_argument("--lime_group_by_node", action="store_true", default=True,
                     help="Group all sub-features of the same ROI node into one interpretable "
                          "unit for LIME (default: on -- keeps it tractable).")
    ap.add_argument("--no_lime_group_by_node", dest="lime_group_by_node", action="store_false")
    ap.add_argument("--kernelshap_group_by_node", action="store_true", default=True,
                     help="Same grouping, for KernelSHAP.")
    ap.add_argument("--no_kernelshap_group_by_node", dest="kernelshap_group_by_node",
                     action="store_false")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                     help=f"Default: {DEFAULT_ROAR_THRESHOLDS}")
    ap.add_argument("--recompute_threshold_zero", action="store_true",
                     help="Sanity-check option: actually retrain at threshold=0.0 instead of "
                          "reusing the baseline run's own metrics (off by default).")
    ap.add_argument("--baseline_only", action="store_true",
                     help="Only reproduce the target fold and save its checkpoint; skip "
                          "captum/ROAR entirely.")
    args = ap.parse_args()

    _add_repo_root_to_path(args.repo_root)

    cudnn.deterministic = True  # matches main.py's `if __name__ == '__main__':` block

    thresholds = args.thresholds if args.thresholds is not None else DEFAULT_ROAR_THRESHOLDS
    current_seed = args.base_seed + (args.target_fold - 1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = Path(args.out_dir)
    baseline_dir = out_dir / "baseline"
    importance_dir = out_dir / "importance"
    history_csv = out_dir / "roar_results_history.csv"
    importance_index_csv = importance_dir / "importance_index.csv"

    print(f"=== run_id={run_id}  target_fold={args.target_fold}  "
          f"current_seed={current_seed}  device={args.device} ===")

    # ---- 1) Reproduce the target fold's exact baseline training ----
    t0 = time.time()
    baseline = reproduce_repeat(args.config_filename, current_seed, args.device)
    print(f"[baseline] trained in {time.time() - t0:.1f}s -- "
          f"test_acc={baseline['metrics'].get('test_acc'):.4f}  "
          f"val_acc={baseline['metrics'].get('val_acc'):.4f}")

    model = baseline["model"]
    config = baseline["config"]
    mvgcn_view_meta = baseline["mvgcn_view_meta"]
    train_dataloader = baseline["dataloaders"][0]

    os.makedirs(baseline_dir, exist_ok=True)
    ckpt_path = baseline_dir / f"baseline_{run_id}.pt"
    torch.save(model.state_dict(), ckpt_path)

    manifest = {
        "run_id": run_id,
        "target_fold": args.target_fold,
        "base_seed": args.base_seed,
        "current_seed": current_seed,
        "config_filename": os.path.abspath(args.config_filename),
        "config": config,
        "metrics": baseline["metrics"],
        "checkpoint_path": str(ckpt_path),
        "device": args.device,
        "timestamp": datetime.now().isoformat(),
        "env": env_info(),
    }
    manifest_path = baseline_dir / f"baseline_manifest_{run_id}.json"
    save_json(manifest_path, manifest)
    print(f"[baseline] saved checkpoint -> {ckpt_path}")
    print(f"[baseline] saved manifest   -> {manifest_path}")

    if args.baseline_only:
        print("--baseline_only set: stopping here.")
        return

    # ---- 2) Column bookkeeping + names (RNG-free, safe to do now) ----
    offset_table = build_offset_table(mvgcn_view_meta)
    view_node_names, extra_names = get_feature_names(config["data"], offset_table)

    def describe_col(col_idx):
        for v in offset_table["view_names"]:
            start = offset_table["view_col_offset"][v]
            n_nodes = offset_table["n_nodes_per_view"][v]
            n_subfeat = offset_table["n_subfeat_per_view"][v]
            span = n_nodes * n_subfeat
            if start <= col_idx < start + span:
                local = col_idx - start
                node_idx, subfeat_idx = divmod(local, n_subfeat)
                roi_name = view_node_names[v][node_idx] if node_idx < len(view_node_names[v]) else f"node_{node_idx}"
                return v, roi_name, node_idx, subfeat_idx
        # extra features
        local = col_idx - offset_table["extra_offset"]
        name = extra_names[local] if 0 <= local < len(extra_names) else f"extra_{local}"
        return "extra", name, local, None

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    all_roar_rows = []
    importance_index_rows = []

    for method in args.methods:
        print(f"\n=== method: {method} ===")
        method_params = {}

        if method == "random":
            ranking_cols = get_random_ranking(offset_table, seed=current_seed)
            score_vector = None
        else:
            method_params = {"target_mode": args.target_class_mode}
            if method == "ig":
                method_params.update({"n_steps": args.ig_steps, "baseline": "zero"})
            elif method == "guided_backprop":
                pass
            elif method == "deeplift":
                method_params.update({"baseline": args.deeplift_baseline})
            elif method == "deeplift_shap":
                method_params.update({"shap_background": args.shap_background,
                                       "n_background_samples": args.n_background_samples})
            elif method == "gradient_shap":
                method_params.update({"shap_background": args.shap_background,
                                       "n_background_samples": args.n_background_samples,
                                       "gradientshap_n_samples": args.gradientshap_n_samples,
                                       "gradientshap_stdev": args.gradientshap_stdev})
            elif method == "lime":
                method_params.update({"n_samples": args.lime_n_samples,
                                       "group_by_node": args.lime_group_by_node})
            elif method == "kernel_shap":
                method_params.update({"n_samples": args.kernelshap_n_samples,
                                       "group_by_node": args.kernelshap_group_by_node})
            else:
                raise ValueError(method)

            score_vector = compute_attributions(
                model, train_dataloader, method, offset_table, device,
                ig_steps=args.ig_steps,
                deeplift_baseline=args.deeplift_baseline,
                target_mode=args.target_class_mode,
                shap_background=args.shap_background,
                n_background_samples=args.n_background_samples,
                gradientshap_n_samples=args.gradientshap_n_samples,
                gradientshap_stdev=args.gradientshap_stdev,
                lime_n_samples=args.lime_n_samples,
                kernelshap_n_samples=args.kernelshap_n_samples,
                lime_group_by_node=args.lime_group_by_node,
                kernelshap_group_by_node=args.kernelshap_group_by_node,
                seed=current_seed,
            )
            ranking_cols = rank_columns_by_score(score_vector)

            # save full importance table for this method/run
            imp_rows = []
            for col_idx in range(offset_table["total_dim"]):
                view, roi_name, node_idx, subfeat_idx = describe_col(col_idx)
                imp_rows.append({
                    "run_id": run_id, "fold": args.target_fold, "method": method,
                    "view": view, "roi_name": roi_name,
                    "node_idx": node_idx, "subfeat_idx": subfeat_idx,
                    "flat_col": col_idx,
                    "importance": float(score_vector[col_idx]),
                })
            imp_df = pd.DataFrame(imp_rows).sort_values("importance", ascending=False)
            imp_path = importance_dir / f"feature_importance_{run_id}_{method}.csv"
            os.makedirs(importance_dir, exist_ok=True)
            imp_df.to_csv(imp_path, index=False)
            print(f"  saved importance table -> {imp_path}")

            importance_index_rows.append({
                "run_id": run_id, "fold": args.target_fold, "method": method,
                "params": json.dumps(method_params), "path": str(imp_path),
                "timestamp": datetime.now().isoformat(),
            })

        roar_rows = run_roar(method, ranking_cols, baseline["metrics"],
                              args.config_filename, current_seed, args.device,
                              thresholds, recompute_threshold_zero=args.recompute_threshold_zero)
        for r in roar_rows:
            r.update({
                "run_id": run_id, "fold": args.target_fold, "base_seed": args.base_seed,
                "current_seed": current_seed, "config_filename": args.config_filename,
                "epochs_per_run": config["train"]["epochs"],
                "method_params": json.dumps(method_params),
                "timestamp": datetime.now().isoformat(),
            })
        all_roar_rows.extend(roar_rows)

    # ---- 3) Persist everything, append-only ----
    if all_roar_rows:
        append_csv(history_csv, pd.DataFrame(all_roar_rows))
        print(f"\n[history] appended {len(all_roar_rows)} row(s) -> {history_csv}")
    if importance_index_rows:
        append_csv(importance_index_csv, pd.DataFrame(importance_index_rows))
        print(f"[history] appended {len(importance_index_rows)} row(s) -> {importance_index_csv}")

    print(f"\nDone. run_id={run_id}")


if __name__ == "__main__":
    main()
