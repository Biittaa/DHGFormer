"""
explainability_gradient.py
---------------------------
Stage-2 explainability for DHGFormer: gradient / attribution based methods.

Stage-1 (explainability.py) works on the *learned* adjacency matrix that the
model produces internally (matrices/learnable_matrix.npy). This stage instead
attributes the model's *decision* back to the raw ROI x ROI functional
connectivity input (the `pearson` / node_features tensor) using Integrated
Gradients (default) or plain vanilla-gradient saliency.

It loads a trained checkpoint (the model_XX%.pt written by
BasicTrain.save_result()) and rebuilds the *exact* test split that was used
for that fold/repeat (same seed, same fold_idx / repeat_idx), so subject-level
alignment between the saliency maps and the labels is correct.

Usage (k-fold run):
    python explainability_gradient.py \
        --config_filename setting/abide_DHGFormer.yaml \
        --checkpoint "result/DHGFormer/ABIDE_cc200/kfold_runs/run_20260821-120000/fold_1/ 85.714%_08-21-12-30-00/model_ 85.714%.pt" \
        --out_dir "result/DHGFormer/ABIDE_cc200/kfold_runs/run_20260821-120000/fold_1/explainability_gradient" \
        --mode kfold --fold_idx 0 --kfold 5 --seed 21

Usage (repeated-run, single split):
    python explainability_gradient.py \
        --config_filename setting/abide_DHGFormer.yaml \
        --checkpoint "result/DHGFormer/ABIDE_cc200/repeated_runs/run_.../repeat_1/.../model_XX%.pt" \
        --out_dir "result/DHGFormer/ABIDE_cc200/repeated_runs/run_.../repeat_1/explainability_gradient" \
        --mode repeat --repeat_idx 0 --seed 21

NOTE: watch the quoting/spaces in --checkpoint if your best_acc formatting
(" {:.3f}%") produced a leading space in the folder/file name.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from model.DHGFormer import DHGFormer
from explainability import (
    SUBNETWORK_ENDS, _subnetwork_slices, load_network_names, load_roi_names,
    guess_node_clus_map_path, rank_networks_by_importance, plot_network_ranking,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# 0. reproducibility helper
# --------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 1. rebuild the exact test split used for this checkpoint
# --------------------------------------------------------------------------

def build_test_dataloader(config, args):
    """Reproduces the same test subjects the checkpoint was evaluated on."""
    if args.mode == "kfold":
        from kfold_dataloader import init_dataloader_kfold
        val_ratio = config["train"].get(
            "val_ratio", config.get("data", {}).get("val_set", 0.1))
        dataloaders, node_size, node_feature_size, timeseries_size, smri_dim = \
            init_dataloader_kfold(
                config["data"], fold_idx=args.fold_idx, kfold=args.kfold,
                val_ratio=val_ratio, seed=args.seed
            )
    else:  # "repeat" -> mirrors main.py's single random_split
        from dataloader import init_dataloader
        current_seed = args.seed + args.repeat_idx
        seed_everything(current_seed)
        dataloaders, node_size, node_feature_size, timeseries_size, smri_dim = \
            init_dataloader(config["data"])

    _, _, test_dataloader = dataloaders
    return test_dataloader, node_size, node_feature_size, timeseries_size, smri_dim


# --------------------------------------------------------------------------
# 2. attribution methods
# --------------------------------------------------------------------------

def integrated_gradients_single(model, time_series, node_features, smri,
                                 target_class, steps=50):
    """
    time_series   : [1, ROI, T]
    node_features : [1, ROI, ROI]  <- the FC input being attributed
    smri          : [1, smri_dim]
    Returns an IG map with the same shape as node_features.
    """
    model.eval()
    baseline = torch.zeros_like(node_features)
    accumulated_grad = torch.zeros_like(node_features)

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = baseline + alpha * (node_features - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)

        output, _, _ = model(time_series, interpolated, smri)
        score = output[0, target_class]

        model.zero_grad()
        grad = torch.autograd.grad(score, interpolated, retain_graph=False)[0]
        accumulated_grad = accumulated_grad + grad

    avg_grad = accumulated_grad / steps
    ig = (node_features - baseline) * avg_grad
    return ig.detach()


def vanilla_saliency_single(model, time_series, node_features, smri, target_class):
    """Plain d(score)/d(input) -- cheap sanity check / fallback."""
    model.eval()
    node_features = node_features.clone().detach().requires_grad_(True)
    output, _, _ = model(time_series, node_features, smri)
    score = output[0, target_class]
    model.zero_grad()
    grad = torch.autograd.grad(score, node_features)[0]
    return grad.detach()


# --------------------------------------------------------------------------
# 3. report: run attribution over the whole test set, aggregate, plot
# --------------------------------------------------------------------------

class GradientExplainabilityReport:
    """Mirrors the structure of explainability.ExplainabilityReport but
    operates on gradient-based saliency instead of the learned adjacency."""

    def __init__(self, model, test_dataloader, out_dir, node_clus_map_path=None,
                 subnetwork_ends=SUBNETWORK_ENDS, method="ig", ig_steps=50,
                 target="predicted"):
        self.model = model.to(device)
        self.test_dataloader = test_dataloader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.subnetwork_ends = subnetwork_ends
        self.method = method              # "ig" or "saliency"
        self.ig_steps = ig_steps
        self.target = target              # "predicted" or "true"

        # Resolve node_clus_map.pickle the same way Stage-1 does: if the
        # caller passed an explicit path use it, otherwise search out_dir
        # and its parents (so it works regardless of CWD).
        resolved_path = Path(node_clus_map_path) if node_clus_map_path else \
            guess_node_clus_map_path(self.out_dir)
        print(f"[GradientExplainabilityReport] using node_clus_map: '{resolved_path}'")

        self.network_names = load_network_names(resolved_path, subnetwork_ends)
        self.roi_names = load_roi_names(resolved_path)

        # Persist what was actually resolved so the Yeo-network mapping is
        # inspectable as data, not just as tick labels baked into a PNG.
        with open(self.out_dir / "network_roi_mapping.json", "w") as f:
            json.dump({
                "node_clus_map_path": str(resolved_path),
                "network_names": list(self.network_names),
                "roi_names": list(self.roi_names) if self.roi_names is not None else None,
                "subnetwork_ends": list(self.subnetwork_ends),
            }, f, indent=2, ensure_ascii=False)

        self.saliency_maps = None
        self.labels = None
        self.predicted = None
        self.classes = None

    # ---- run attribution over every test subject ----
    def compute(self):
        saliency_list, label_list, pred_list = [], [], []

        for data_in, pearson, label, _, smri in self.test_dataloader:
            label = label.long()
            data_in, pearson, label, smri = (
                data_in.to(device), pearson.to(device),
                label.to(device), smri.to(device)
            )
            batch_size = data_in.shape[0]

            with torch.no_grad():
                logits, _, _ = self.model(data_in, pearson, smri)
                predicted_cls = logits.argmax(dim=1)

            for b in range(batch_size):
                ts_b = data_in[b:b + 1]
                fc_b = pearson[b:b + 1]
                smri_b = smri[b:b + 1]
                target_cls = int(
                    predicted_cls[b].item() if self.target == "predicted"
                    else label[b].item()
                )

                if self.method == "ig":
                    sal = integrated_gradients_single(
                        self.model, ts_b, fc_b, smri_b, target_cls,
                        steps=self.ig_steps
                    )
                else:
                    sal = vanilla_saliency_single(
                        self.model, ts_b, fc_b, smri_b, target_cls
                    )

                saliency_list.append(sal.squeeze(0).cpu().numpy())
                label_list.append(int(label[b].item()))
                pred_list.append(int(predicted_cls[b].item()))

        self.saliency_maps = np.stack(saliency_list)   # [N, ROI, ROI]
        self.labels = np.array(label_list)
        self.predicted = np.array(pred_list)
        self.classes = sorted(np.unique(self.labels).tolist())

        np.savez(self.out_dir / f"{self.method}_saliency_raw.npz",
                 saliency=self.saliency_maps, label=self.labels,
                 predicted=self.predicted)
        return self.saliency_maps

    # ---- 1. average |saliency| per class ----
    def plot_group_average(self):
        avgs = {c: np.abs(self.saliency_maps[self.labels == c]).mean(axis=0)
                for c in self.classes}
        fig, axes = plt.subplots(1, len(avgs), figsize=(6 * len(avgs), 5))
        if len(avgs) == 1:
            axes = [axes]
        for ax, (c, mat) in zip(axes, avgs.items()):
            sns.heatmap(mat, ax=ax, cmap="viridis",
                        cbar_kws={"label": f"|{self.method}| saliency"})
            ax.set_title(f"Class {c} (n={int(np.sum(self.labels == c))})")
        plt.tight_layout()
        fig.savefig(self.out_dir / f"{self.method}_group_average.png", dpi=200)
        plt.close(fig)
        return avgs

    # ---- 2. group difference (which edges push toward class 1 vs class 0) ----
    def group_difference(self):
        assert len(self.classes) == 2, "expects a binary task"
        c0, c1 = self.classes
        s0 = self.saliency_maps[self.labels == c0]
        s1 = self.saliency_maps[self.labels == c1]
        diff = s1.mean(axis=0) - s0.mean(axis=0)

        t_vals, p_vals = stats.ttest_ind(s1, s0, axis=0, equal_var=False)
        p_vals = np.nan_to_num(p_vals, nan=1.0)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.heatmap(diff, ax=axes[0], cmap="RdBu_r", center=0,
                    cbar_kws={"label": f"class{c1} - class{c0} saliency"})
        axes[0].set_title(f"{self.method.upper()} group difference")

        sig_diff = np.where(p_vals >= 0.05, 0, diff)
        sns.heatmap(sig_diff, ax=axes[1], cmap="RdBu_r", center=0,
                    cbar_kws={"label": "significant edges (p<0.05)"})
        axes[1].set_title("Significant edges (uncorrected t-test)")
        plt.tight_layout()
        fig.savefig(self.out_dir / f"{self.method}_group_difference.png", dpi=200)
        plt.close(fig)

        np.savez(self.out_dir / f"{self.method}_group_difference.npz",
                 diff=diff, t_vals=t_vals, p_vals=p_vals)
        return diff, t_vals, p_vals

    # ---- 3. subnetwork-level saliency summary ----
    def subnetwork_summary(self):
        slices = _subnetwork_slices(self.subnetwork_ends)
        n = len(slices)
        avgs = {c: np.abs(self.saliency_maps[self.labels == c]).mean(axis=0)
                for c in self.classes}
        result = {c: np.zeros((n, n)) for c in self.classes}
        for c, mat in avgs.items():
            for i, (si, ei) in enumerate(slices):
                for j, (sj, ej) in enumerate(slices):
                    result[c][i, j] = mat[si:ei, sj:ej].mean()

        fig, axes = plt.subplots(1, len(result), figsize=(6 * len(result), 5))
        if len(result) == 1:
            axes = [axes]
        for ax, (c, mat) in zip(axes, result.items()):
            sns.heatmap(mat, ax=ax, cmap="viridis",
                        xticklabels=self.network_names, yticklabels=self.network_names,
                        annot=True, fmt=".2f")
            ax.set_title(f"Subnetwork {self.method.upper()} saliency - class {c}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(self.out_dir / f"{self.method}_subnetwork_summary.png", dpi=200)
        plt.close(fig)

        # persist the numbers + labels, not just the plot
        np.savez(self.out_dir / f"{self.method}_subnetwork_summary.npz",
                 **{f"class_{c}": mat for c, mat in result.items()},
                 network_names=np.array(self.network_names, dtype=object))
        with open(self.out_dir / f"{self.method}_subnetwork_summary.json", "w") as f:
            json.dump({
                "network_names": list(self.network_names),
                "matrices": {str(c): mat.tolist() for c, mat in result.items()}
            }, f, indent=2, ensure_ascii=False)
        return result

    # ---- 3b. Yeo-network-level importance ranking ----
    def network_ranking(self, top_k=20):
        """Which of the 8 Yeo(+subcortical) networks matter most, aggregating
        the same per-ROI |group-difference of saliency| score used in
        roi_ranking() up to network level. See
        explainability.rank_networks_by_importance() for the criteria."""
        diff, _, _ = self.group_difference()
        node_score = np.abs(diff).sum(axis=1)
        ranking = rank_networks_by_importance(
            node_score, self.subnetwork_ends, self.network_names, top_k=top_k)

        plot_network_ranking(
            ranking, self.out_dir / f"{self.method}_network_ranking.png",
            title=f"Yeo network importance ranking ({self.method.upper()} saliency)")

        with open(self.out_dir / f"{self.method}_network_ranking.json", "w") as f:
            json.dump(ranking, f, indent=2, ensure_ascii=False)
        return ranking

    # ---- 4. discriminative ROI ranking ----
    def roi_ranking(self, top_k=20):
        diff, _, _ = self.group_difference()
        node_score = np.abs(diff).sum(axis=1)
        order = np.argsort(-node_score)[:top_k]

        labels = (self.roi_names if self.roi_names is not None
                  else [f"ROI {i}" for i in range(len(node_score))])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([labels[i] for i in order][::-1], node_score[order][::-1])
        ax.set_xlabel(f"Sum |{self.method} delta saliency|")
        ax.set_title(f"Top-{top_k} discriminative ROIs ({self.method.upper()})")
        plt.tight_layout()
        fig.savefig(self.out_dir / f"{self.method}_roi_ranking.png", dpi=200)
        plt.close(fig)

        ranking = [{"roi_index": int(i), "roi_name": labels[i], "score": float(node_score[i])}
                   for i in order]
        with open(self.out_dir / f"{self.method}_roi_ranking.json", "w") as f:
            json.dump(ranking, f, indent=2)
        return ranking

    # ---- 5. connectogram of top salient edges ----
    def connectogram(self, top_edges=60):
        diff, _, _ = self.group_difference()
        n = diff.shape[0]
        iu = np.triu_indices(n, k=1)
        strength = np.abs(diff[iu])
        order = np.argsort(-strength)[:top_edges]
        sel_i, sel_j, sel_w = iu[0][order], iu[1][order], diff[iu][order]

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = np.cos(angles), np.sin(angles)
        max_w = np.max(np.abs(sel_w)) + 1e-8

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(xs, ys, s=20, c="black", zorder=3)
        for i, j, w in zip(sel_i, sel_j, sel_w):
            color = "crimson" if w > 0 else "royalblue"
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=color,
                    alpha=min(1.0, abs(w) / max_w), linewidth=1.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Top-{top_edges} {self.method.upper()}-discriminative edges\n"
                     "(red = pushes toward class 1, blue = pushes toward class 0)")
        fig.savefig(self.out_dir / f"{self.method}_connectogram.png", dpi=200)
        plt.close(fig)

    def run_all(self):
        self.compute()
        self.plot_group_average()
        self.group_difference()
        self.subnetwork_summary()
        self.roi_ranking()
        self.network_ranking()
        self.connectogram()
        acc = float(np.mean(self.predicted == self.labels))
        print(f"[explainability_gradient] test-set accuracy while attributing: {acc:.3f}")
        print(f"[explainability_gradient] figures saved to: {self.out_dir}")


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default="setting/abide_DHGFormer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["kfold", "repeat"], default="kfold")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--repeat_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21,
                        help="the base seed the run was launched with")
    parser.add_argument("--method", choices=["ig", "saliency"], default="ig")
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--target", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--node_clus_map", default=None,
                        help="Path to node_clus_map.pickle. If omitted, the "
                             "script searches out_dir and its parent "
                             "directories for it (same logic as Stage-1).")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    seed_everything(args.seed)
    test_dataloader, node_size, node_feature_size, timeseries_size, smri_dim = \
        build_test_dataloader(config, args)

    model = DHGFormer(
        config["model"], node_size, node_feature_size, timeseries_size,
        use_smri=config["data"].get("use_smri", False), smri_input_dim=smri_dim
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    report = GradientExplainabilityReport(
        model, test_dataloader, args.out_dir, node_clus_map_path=args.node_clus_map,
        method=args.method, ig_steps=args.ig_steps, target=args.target
    )
    report.run_all()


if __name__ == "__main__":
    main()
