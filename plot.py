"""
Plot FID vs. number of function evaluations (NFE) for every results/*.json
produced by eval.py, comparing flow-matching (multi-step Euler) against
mean-flow (1-4+ step) checkpoints on the same axes.

Usage:
    python plot_results.py
    python plot_results.py --results_dir results --out fid_comparison.png
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt


def load_results(results_dir, filenames=None):
    """Returns a dict: {loss_type: {checkpoint_stem: {step: fid}}}

    If `filenames` is given, only those files (relative to results_dir, with
    or without the .json extension) are loaded instead of every json present.
    """
    if filenames:
        paths = []
        for name in filenames:
            if not name.endswith(".json"):
                name += ".json"
            paths.append(os.path.join(results_dir, name))
    else:
        paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))

    grouped = {}
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        loss_type = data["config"]["training"].get("loss_type", "flow_matching")
        stem = os.path.splitext(os.path.basename(path))[0]
        fid_map = {int(k): v for k, v in data["fid"].items()}

        grouped.setdefault(loss_type, {})[stem] = fid_map
    return grouped


def plot(grouped, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"flow_matching": "tab:blue", "mean_flow": "tab:red"}
    markers = {"flow_matching": "o", "mean_flow": "s"}

    for loss_type, checkpoints in grouped.items():
        color = colors.get(loss_type, "tab:gray")
        marker = markers.get(loss_type, "x")
        for stem, fid_map in checkpoints.items():
            steps = sorted(fid_map.keys())
            fids = [fid_map[s] for s in steps]
            ax.plot(
                steps, fids,
                marker=marker, color=color, linewidth=2, markersize=7,
                label=f"{loss_type} ({stem})",
            )

    ax.set_xlabel("Number of function evaluations (NFE)")
    ax.set_ylabel("FID (lower is better)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_title("FID vs. sampling steps: flow matching vs. mean flow")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out", type=str, default="fid_comparison.png")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                         help="Specific result filenames (with or without .json) to plot, "
                              "e.g. --files unet_mnist_large_v3_epoch_600 unet_mnist_large_mf_v2_epoch_400. "
                              "If omitted, every *.json in results_dir is plotted.")
    args = parser.parse_args()

    grouped = load_results(args.results_dir, filenames=args.files)
    if not grouped:
        print(f"No results found in {args.results_dir}/*.json")
    else:
        plot(grouped, args.out)