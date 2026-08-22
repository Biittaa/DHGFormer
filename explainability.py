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

# This project's node_clus_map.pickle is the same file used in the
# Com-BrainTF repo (ubc-tea/Com-BrainTF, Bannadabhavi et al., MICCAI 2023) --
# verified byte-for-byte identical. Its dict values are integer community ids
# (0..7), not strings. That paper (Sec 3.1) states the eight communities are,
# in this exact order: cerebellum/subcortical, visual, somatomotor, dorsal
# attention, ventral attention, limbic, frontoparietal, DMN -- matching the
# same order shown in the community-id -> ROI-count signature below (community
# id N has KNOWN_CLUSTER_SIZE_SIGNATURE[N] ROIs). We only apply these names
# when the sizes match exactly, so an unrelated node_clus_map.pickle won't
# silently get mislabeled.
KNOWN_CLUSTER_SIZE_SIGNATURE = [41, 29, 21, 19, 20, 7, 21, 42]
KNOWN_CLUSTER_NAMES = [
    "Subcortical-Cerebellum", "Visual", "Somatomotor", "DorsalAttention",
    "VentralAttention", "Limbic", "Frontoparietal", "DMN",
]


def _subnetwork_slices(subnetwork_ends):
    starts = [0] + subnetwork_ends[:-1]
    return list(zip(starts, subnetwork_ends))


def rank_networks_by_importance(node_score, subnetwork_ends, network_names, top_k=20):
    """Aggregate per-ROI importance scores (e.g. |group-difference| row-sum,
    or |saliency| node strength) up to the Yeo-network/subnetwork level, using
    the aggregation strategies most commonly reported in the brain-network
    literature (e.g. BrainGNN, Com-BrainTF, BrainNetTransformer, FBNetGen):

      - mean_importance: average per-ROI importance *within* the network.
        This is the primary, size-normalized ranking -- it does not
        automatically favor a larger network (DMN, 42 ROIs here) over a
        smaller one (Limbic, 7 ROIs) just because it has more nodes, which
        a raw sum would.
      - sum_importance: total importance mass contributed by the network
        (does scale with network size; useful for "how much of the overall
        signal comes from here").
      - topk_count / topk_fraction: how many of the global top-`top_k` most
        important ROIs fall inside this network. This is the metric papers
        typically use informally when they report findings as "mainly
        localizing to network X" (e.g. DHGFormer's own Fig. 2(b) discussion
        of DMN-localized discriminative ROIs).

    Returns a list of one dict per network, sorted by descending
    mean_importance (the primary/default ranking), each dict also carrying
    its rank under every metric so you can cross-check which criterion you
    prefer.
    """
    slices = _subnetwork_slices(subnetwork_ends)
    node_score = np.asarray(node_score)
    top_k = min(top_k, len(node_score))
    order_topk = np.argsort(-node_score)[:top_k]

    rows = []
    for idx, (s, e) in enumerate(slices):
        block = node_score[s:e]
        topk_count = int(np.sum((order_topk >= s) & (order_topk < e)))
        rows.append({
            "network": network_names[idx],
            "num_rois": int(e - s),
            "mean_importance": float(block.mean()),
            "sum_importance": float(block.sum()),
            "max_importance": float(block.max()),
            "topk_count": topk_count,
            "topk_fraction": topk_count / top_k,
        })

    for metric, key in [("mean_importance", "rank_by_mean"),
                        ("sum_importance", "rank_by_sum"),
                        ("topk_count", "rank_by_topk")]:
        for rank, r in enumerate(sorted(rows, key=lambda r: -r[metric]), 1):
            r[key] = rank

    rows.sort(key=lambda r: -r["mean_importance"])
    return rows


def plot_network_ranking(ranking, out_path, title, metric="mean_importance",
                          metric_label="Mean |importance| per ROI (size-normalized)"):
    """Bar chart of network_ranking() output, sorted by `metric`."""
    ordered = sorted(ranking, key=lambda r: -r[metric])
    names = [r["network"] for r in ordered]
    values = [r[metric] for r in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_fold_matrices(matrices_path: Path):
    """matrices_path points directly at learnable_matrix.npy"""
    data = np.load(matrices_path, allow_pickle=True).item()
    matrices = np.asarray(data["matrix"])              # [N, ROI, ROI]
    labels = np.asarray(data["label"]).astype(int)      # [N]
    return matrices, labels


def _cluster_id_names(values, node_clus_map_path):
    """values: list of int community ids (0..K-1), in node_clus_map.pickle's
    dict-iteration order (== cluster_order used by DHGFormer.reorder_nodes).
    Returns a name per community id: the known Yeo+subcortical names if the
    per-community ROI counts match KNOWN_CLUSTER_SIZE_SIGNATURE exactly,
    otherwise generic 'Cluster-{id}' labels (never a misleading Net-i guess)."""
    from collections import Counter
    counts = Counter(values)
    num_clusters = max(counts) + 1
    sizes = [counts.get(i, 0) for i in range(num_clusters)]
    if sizes == KNOWN_CLUSTER_SIZE_SIGNATURE:
        print(f"[load_network_names] '{node_clus_map_path}' matches the known "
              f"Com-BrainTF/DHGFormer 8-community cc200 signature {sizes}; "
              f"using verified names {KNOWN_CLUSTER_NAMES} "
              f"(source: Bannadabhavi et al., MICCAI 2023, Sec 3.1).")
        return KNOWN_CLUSTER_NAMES
    print(f"[load_network_names] '{node_clus_map_path}' has integer community "
          f"ids with sizes {sizes}, which does NOT match the known signature "
          f"{KNOWN_CLUSTER_SIZE_SIGNATURE} -- using generic 'Cluster-{{id}}' "
          f"names instead of guessing Yeo network identities.")
    return [f"Cluster-{i}" for i in range(num_clusters)]


def load_network_names(node_clus_map_path: Path, subnetwork_ends=SUBNETWORK_ENDS):
    """Best-effort: derive a readable name per subnetwork block from
    node_clus_map.pickle. Supports two value formats:
      - string values (the network name is stored directly per ROI), or
      - integer community-id values (0..K-1), in which case names are
        resolved via _cluster_id_names (see KNOWN_CLUSTER_SIZE_SIGNATURE).
    Falls back to generic Net-1..Net-8 names on any failure.

    NOTE: any failure here (missing file, wrong pickle structure, unexpected
    value type, length mismatch with subnetwork_ends) is reported with a
    [load_network_names] warning instead of being swallowed silently, so you
    can tell *why* you're getting fallback names instead of real ones.
    """
    node_clus_map_path = Path(node_clus_map_path) if node_clus_map_path else None
    if node_clus_map_path is None or not Path(node_clus_map_path).exists():
        print(f"[load_network_names] WARNING: node_clus_map path not found: "
              f"'{node_clus_map_path}'. Falling back to {DEFAULT_NETWORK_NAMES}.")
        return DEFAULT_NETWORK_NAMES
    try:
        with open(node_clus_map_path, "rb") as f:
            node_cluster_map = pickle.load(f)
        values = list(node_cluster_map.values())
        if not values:
            print(f"[load_network_names] WARNING: '{node_clus_map_path}' loaded "
                  f"but is empty. Falling back to {DEFAULT_NETWORK_NAMES}.")
            return DEFAULT_NETWORK_NAMES

        if isinstance(values[0], str):
            if len(values) < subnetwork_ends[-1]:
                print(f"[load_network_names] WARNING: '{node_clus_map_path}' has "
                      f"{len(values)} entries but subnetwork_ends expects "
                      f"{subnetwork_ends[-1]}. Falling back to {DEFAULT_NETWORK_NAMES}.")
                return DEFAULT_NETWORK_NAMES
            names = []
            for s, e in _subnetwork_slices(subnetwork_ends):
                block = values[s:e]
                names.append(max(set(block), key=block.count))
            return names

        if isinstance(values[0], (int, np.integer)):
            return _cluster_id_names(values, node_clus_map_path)

        print(f"[load_network_names] WARNING: values in '{node_clus_map_path}' "
              f"are of type {type(values[0])}, neither str nor int. "
              f"Falling back to {DEFAULT_NETWORK_NAMES}.")
        return DEFAULT_NETWORK_NAMES
    except Exception as e:
        print(f"[load_network_names] WARNING: failed to read "
              f"'{node_clus_map_path}' ({type(e).__name__}: {e}). "
              f"Falling back to {DEFAULT_NETWORK_NAMES}.")
        return DEFAULT_NETWORK_NAMES


def load_roi_names(node_clus_map_path: Path):
    """Per-ROI label = the community/network each ROI belongs to, in the same
    order the model/explainability index ROIs (post cluster_order reorder).
    Supports both string-valued and integer-community-id-valued pickles.
    Returns None if it can't be derived (caller falls back to 'ROI {i}')."""
    node_clus_map_path = Path(node_clus_map_path) if node_clus_map_path else None
    if node_clus_map_path is None or not Path(node_clus_map_path).exists():
        print(f"[load_roi_names] WARNING: node_clus_map path not found: "
              f"'{node_clus_map_path}'. ROI labels will fall back to 'ROI {{i}}'.")
        return None
    try:
        with open(node_clus_map_path, "rb") as f:
            node_cluster_map = pickle.load(f)
        keys = list(node_cluster_map.keys())
        values = list(node_cluster_map.values())
        if values and isinstance(values[0], str):
            return [f"ROI{i}-{values[i]}" for i in range(len(values))]
        if values and isinstance(values[0], (int, np.integer)):
            cluster_names = _cluster_id_names(values, node_clus_map_path)
            return [f"ROI{i}(orig{keys[i]})-{cluster_names[values[i]]}"
                    for i in range(len(values))]
        print(f"[load_roi_names] WARNING: values in '{node_clus_map_path}' are "
              f"of type {type(values[0]) if values else None}, neither str "
              f"nor int. ROI labels will fall back to 'ROI {{i}}'.")
    except Exception as e:
        print(f"[load_roi_names] WARNING: failed to read '{node_clus_map_path}' "
              f"({type(e).__name__}: {e}). ROI labels will fall back to 'ROI {{i}}'.")
    return None


def guess_node_clus_map_path(start_dir):
    """Search start_dir and its parents for node_clus_map.pickle. Shared by
    ExplainabilityReport (Stage-1) and GradientExplainabilityReport (Stage-2)
    so both resolve the same file the same way regardless of CWD."""
    start_dir = Path(start_dir)
    for parent in [start_dir, *start_dir.parents]:
        candidate = parent / "node_clus_map.pickle"
        if candidate.exists():
            return candidate
    return Path("node_clus_map.pickle")


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
        
        
        guess_path = node_clus_map_path or self._guess_node_clus_map_path()
        print(f"[ExplainabilityReport] using node_clus_map: '{guess_path}'")
        if network_names is not None:
            self.network_names = network_names
        else:
            self.network_names = load_network_names(guess_path, subnetwork_ends)
        self.roi_names = load_roi_names(guess_path)

        # persist what was actually resolved, so it's inspectable outside the plots
        with open(self.out_dir / "network_roi_mapping.json", "w") as f:
            json.dump({
                "node_clus_map_path": str(guess_path),
                "network_names": list(self.network_names),
                "roi_names": list(self.roi_names) if self.roi_names is not None else None,
                "subnetwork_ends": list(self.subnetwork_ends),
            }, f, indent=2, ensure_ascii=False)

        # if network_names is not None:
        #     self.network_names = network_names
        # else:
        #     guess_path = node_clus_map_path or self._guess_node_clus_map_path()
        #     self.network_names = load_network_names(guess_path, subnetwork_ends)

    def _guess_node_clus_map_path(self):
        # node_clus_map.pickle sits at the repo root, several levels above
        # .../log_folder/model_type/dataset_atlas/run_.../fold_N
        return guess_node_clus_map_path(self.fold_dir)

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

        np.savez(self.out_dir / "subnetwork_summary.npz",
                 **{f"class_{c}": mat for c, mat in result.items()},
                 network_names=np.array(self.network_names, dtype=object))
        with open(self.out_dir / "subnetwork_summary.json", "w") as f:
            json.dump({
                "network_names": list(self.network_names),
                "matrices": {str(c): mat.tolist() for c, mat in result.items()}
            }, f, indent=2, ensure_ascii=False)
        return result

    # ---------- 4. discriminative ROI ranking ----------
    def roi_ranking(self, top_k=20):
        diff, _, _ = self.group_difference()
        node_score = np.abs(diff).sum(axis=1)
        order = np.argsort(-node_score)[:top_k]

        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.barh([f"ROI {i}" for i in order][::-1], node_score[order][::-1])
        labels = (self.roi_names if self.roi_names is not None
                  else [f"ROI {i}" for i in range(len(node_score))])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([labels[i] for i in order][::-1], node_score[order][::-1])
        ax.set_xlabel("Sum |delta connectivity|")
        ax.set_title(f"Top-{top_k} discriminative ROIs")
        plt.tight_layout()
        fig.savefig(self.out_dir / "roi_ranking.png", dpi=200)
        plt.close(fig)

        # ranking = [{"roi_index": int(i), "score": float(node_score[i])} for i in order]
        ranking = [{"roi_index": int(i), "roi_name": labels[i], "score": float(node_score[i])} for i in order]
        with open(self.out_dir / "roi_ranking.json", "w") as f:
            json.dump(ranking, f, indent=2)
        return ranking

    # ---------- 4b. Yeo-network-level importance ranking ----------
    def network_ranking(self, top_k=20):
        """Which of the 8 Yeo(+subcortical) networks matter most, aggregating
        the same per-ROI |group-difference| score used in roi_ranking() up to
        network level. See rank_networks_by_importance() for the criteria."""
        diff, _, _ = self.group_difference()
        node_score = np.abs(diff).sum(axis=1)
        ranking = rank_networks_by_importance(
            node_score, self.subnetwork_ends, self.network_names, top_k=top_k)

        plot_network_ranking(
            ranking, self.out_dir / "network_ranking.png",
            title=f"Yeo network importance ranking (mean |delta connectivity| per ROI)")

        with open(self.out_dir / "network_ranking.json", "w") as f:
            json.dump(ranking, f, indent=2, ensure_ascii=False)
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
        self.network_ranking()
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
