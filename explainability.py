"""
explainability.py
------------------
Stage-1 explainability for DHGFormer: works directly on the learnable
adjacency matrices already produced by BasicTrain.generate_save_learnable_matrix()
(saved as matrices/learnable_matrix.npy -> {'matrix': [N_subj, ROI, ROI], 'label': [N_subj]}).

Stage-2 (gradient / attribution based: saliency on X_ts / X_fc, attention
rollout from the FC-inspired encoder) will live in explainability_gradient.py
and reuse the same ExplainabilityReport.out_dir convention.

Called automatically at the end of BasicTrain.train() (see integration note
in chat), or standalone:

    python explainability.py --fold_dir result/DHGFormer/ABIDE_cc200/run_.../fold_1
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Fixed by the architecture: cc200 atlas reordered into 8 Yeo-based subnetworks
# (must match DHGFormer.subnetwork_ends in model/DHGFormer.py).
SUBNETWORK_ENDS = [41, 70, 91, 110, 130, 137, 158, 200]
DEFAULT_NETWORK_NAMES = [f"Net-{i + 1}" for i in range(len(SUBNETWORK_ENDS))]


def _subnetwork_slices(subnetwork_ends):
    starts = [0] + subnetwork_ends[:-1]
    return list(zip(starts, subnetwork_ends))


def load_fold_matrices(matrices_path: Path):
    """matrices_path points directly at learnable_matrix.npy"""
    data = np.load(matrices_path, allow_pickle=True).item()
    matrices = np.asarray(data["matrix"])              # [N, ROI, ROI]
    labels = np.asarray(data["label"]).astype(int)      # [N]
    return matrices, labels


def load_network_names(node_clus_map_path: Path, subnetwork_ends=SUBNETWORK_ENDS):
    """Best-effort: derive a readable name per subnetwork block from
    node_clus_map.pickle if it stores per-ROI network labels; otherwise
    fall back to generic Net-1..Net-8 names."""
    try:
        with open(node_clus_map_path, "rb") as f:
            node_cluster_map = pickle.load(f)
        values = list(node_cluster_map.values())
        if values and isinstance(values[0], str):
            names = []
            for s, e in _subnetwork_slices(subnetwork_ends):
                block = values[s:e]
                names.append(max(set(block), key=block.count))
            return names
    except Exception:
        pass
    return DEFAULT_NETWORK_NAMES


class ExplainabilityReport:
    """
    fold_dir: the per-fold root, e.g. .../run_20260821-153000/fold_1
    Expects fold_dir/matrices/learnable_matrix.npy to exist (written by
    BasicTrain.generate_save_learnable_matrix()). Writes everything into
    fold_dir/explainability/.
    """

    def __init__(self, fold_dir, node_clus_map_path=None,
                 network_names=None, subnetwork_ends=SUBNETWORK_ENDS):
        self.fold_dir = Path(fold_dir)
        self.matrices_path = self.fold_dir / "matrices" / "learnable_matrix.npy"
        self.out_dir = self.fold_dir / "explainability"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.matrices, self.labels = load_fold_matrices(self.matrices_path)
        self.subnetwork_ends = subnetwork_ends
        self.roi_num = self.matrices.shape[-1]
        self.classes = sorted(np.unique(self.labels).tolist())

        if network_names is not None:
            self.network_names = network_names
        else:
            guess_path = node_clus_map_path or self._guess_node_clus_map_path()
            self.network_names = load_network_names(guess_path, subnetwork_ends)

    def _guess_node_clus_map_path(self):
        # node_clus_map.pickle sits at the repo root, several levels above
        # .../log_folder/model_type/dataset_atlas/run_.../fold_N
        for parent in self.fold_dir.parents:
            candidate = parent / "node_clus_map.pickle"
            if candidate.exists():
                return candidate
        return Path("node_clus_map.pickle")

    # ---------- 1. per-class average connectome ----------
    def group_average_matrices(self):
        return {c: self.matrices[self.labels == c].mean(axis=0) for c in self.classes}

    def plot_group_average(self):
        avgs = self.group_average_matrices()
        fig, axes = plt.subplots(1, len(avgs), figsize=(6 * len(avgs), 5))
        if len(avgs) == 1:
            axes = [axes]
        for ax, (c, mat) in zip(axes, avgs.items()):
            sns.heatmap(mat, ax=ax, cmap="coolwarm", center=0,
                        cbar_kws={"label": "connectivity"})
            ax.set_title(f"Class {c} (n={int(np.sum(self.labels == c))})")
        plt.tight_layout()
        fig.savefig(self.out_dir / "group_average_connectome.png", dpi=200)
        plt.close(fig)
        return avgs

    # ---------- 2. group difference + significance ----------
    def group_difference(self):
        assert len(self.classes) == 2, "group_difference expects a binary task"
        c0, c1 = self.classes
        m0 = self.matrices[self.labels == c0]
        m1 = self.matrices[self.labels == c1]
        diff = m1.mean(axis=0) - m0.mean(axis=0)

        t_vals, p_vals = stats.ttest_ind(m1, m0, axis=0, equal_var=False)
        p_vals = np.nan_to_num(p_vals, nan=1.0)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.heatmap(diff, ax=axes[0], cmap="RdBu_r", center=0,
                    cbar_kws={"label": f"class{c1} - class{c0}"})
        axes[0].set_title("Group difference (mean)")

        sig_diff = np.where(p_vals >= 0.05, 0, diff)
        sns.heatmap(sig_diff, ax=axes[1], cmap="RdBu_r", center=0,
                    cbar_kws={"label": "significant edges only (p<0.05)"})
        axes[1].set_title("Significant edges (uncorrected t-test)")
        plt.tight_layout()
        fig.savefig(self.out_dir / "group_difference.png", dpi=200)
        plt.close(fig)

        np.savez(self.out_dir / "group_difference.npz",
                 diff=diff, t_vals=t_vals, p_vals=p_vals)
        return diff, t_vals, p_vals

    # ---------- 3. subnetwork-level (intra/inter) summary ----------
    def subnetwork_summary(self):
        slices = _subnetwork_slices(self.subnetwork_ends)
        n = len(slices)
        avgs = self.group_average_matrices()
        result = {c: np.zeros((n, n)) for c in self.classes}
        for c, mat in avgs.items():
            for i, (si, ei) in enumerate(slices):
                for j, (sj, ej) in enumerate(slices):
                    result[c][i, j] = mat[si:ei, sj:ej].mean()

        fig, axes = plt.subplots(1, len(result), figsize=(6 * len(result), 5))
        if len(result) == 1:
            axes = [axes]
        for ax, (c, mat) in zip(axes, result.items()):
            sns.heatmap(mat, ax=ax, cmap="coolwarm", center=0,
                        xticklabels=self.network_names, yticklabels=self.network_names,
                        annot=True, fmt=".2f")
            ax.set_title(f"Subnetwork connectivity - class {c}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(self.out_dir / "subnetwork_summary.png", dpi=200)
        plt.close(fig)
        return result

    # ---------- 4. discriminative ROI ranking ----------
    def roi_ranking(self, top_k=20):
        diff, _, _ = self.group_difference()
        node_score = np.abs(diff).sum(axis=1)
        order = np.argsort(-node_score)[:top_k]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([f"ROI {i}" for i in order][::-1], node_score[order][::-1])
        ax.set_xlabel("Sum |delta connectivity|")
        ax.set_title(f"Top-{top_k} discriminative ROIs")
        plt.tight_layout()
        fig.savefig(self.out_dir / "roi_ranking.png", dpi=200)
        plt.close(fig)

        ranking = [{"roi_index": int(i), "score": float(node_score[i])} for i in order]
        with open(self.out_dir / "roi_ranking.json", "w") as f:
            json.dump(ranking, f, indent=2)
        return ranking

    # ---------- 5. connectogram of top discriminative edges ----------
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
        ax.set_title(f"Top-{top_edges} group-discriminative edges\n"
                     "(red = higher in class 1, blue = higher in class 0)")
        fig.savefig(self.out_dir / "connectogram.png", dpi=200)
        plt.close(fig)

    def run_all(self):
        self.plot_group_average()
        self.group_difference()
        self.subnetwork_summary()
        self.roi_ranking()
        self.connectogram()
        print(f"[explainability] figures saved to: {self.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="e.g. result/DHGFormer/ABIDE_cc200/run_.../fold_1")
    parser.add_argument("--node_clus_map", type=str, default=None)
    args = parser.parse_args()
    report = ExplainabilityReport(args.fold_dir, node_clus_map_path=args.node_clus_map)
    report.run_all()
