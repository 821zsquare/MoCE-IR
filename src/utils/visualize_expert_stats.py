import argparse
import csv
import math
import os
import re
from collections import OrderedDict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "active_fraction": "Active fraction among selected experts",
    "activation_probability": "Selection probability per image",
    "gate_mass_per_image": "Gate mass per image",
    "mean_gate_when_active": "Mean gate when active",
}


def natural_layer_key(layer_name):
    match = re.search(r"decoder_stage_(\d+)_block_(\d+)", layer_name)
    if match:
        return int(match.group(1)), int(match.group(2)), layer_name
    numbers = [int(x) for x in re.findall(r"\d+", layer_name)]
    return (*numbers, layer_name)


def sanitize_name(name):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("_") or "unknown"


def read_stats(csv_path, metric):
    data = OrderedDict()
    layers = set()
    experts = set()
    degradations = set()

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"scope", "degradation", "layer", "expert", metric}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            degradation = row["degradation"] if row["scope"] == "degradation" else "ALL"
            layer = row["layer"]
            expert = int(row["expert"])
            value = float(row[metric])

            data.setdefault(degradation, OrderedDict()).setdefault(layer, {})[expert] = value
            layers.add(layer)
            experts.add(expert)
            degradations.add(degradation)

    sorted_layers = sorted(layers, key=natural_layer_key)
    sorted_experts = sorted(experts)
    sorted_degradations = ["ALL"] + sorted(d for d in degradations if d != "ALL")
    return data, sorted_layers, sorted_experts, sorted_degradations


def matrix_for(data, degradation, layers, experts):
    matrix = np.full((len(layers), len(experts)), np.nan, dtype=np.float64)
    layer_data = data.get(degradation, {})
    for row_idx, layer in enumerate(layers):
        expert_values = layer_data.get(layer, {})
        for col_idx, expert in enumerate(experts):
            if expert in expert_values:
                matrix[row_idx, col_idx] = expert_values[expert]
    return matrix


def finite_values_from(matrices):
    values = [matrix[~np.isnan(matrix)] for matrix in matrices if np.any(~np.isnan(matrix))]
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate(values)


def set_common_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def plot_heatmap(matrix, layers, experts, title, metric_label, output_path, vmin, vmax, dpi):
    height = max(4.0, 0.35 * len(layers) + 1.6)
    width = max(5.0, 0.7 * len(experts) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Decoder layer")
    ax.set_xticks(np.arange(len(experts)))
    ax.set_xticklabels([str(expert) for expert in experts])
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isnan(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white" if value > (vmin + vmax) / 2 else "black", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(metric_label)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_combined_degradation_heatmaps(data, layers, experts, degradations, metric, output_path, dpi):
    shown_degradations = [d for d in degradations if d != "ALL"]
    if not shown_degradations:
        return None

    cols = min(3, len(shown_degradations))
    rows = math.ceil(len(shown_degradations) / cols)
    fig_width = max(5.0 * cols, 5.0)
    fig_height = max(3.6 * rows, 3.6)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    matrices = [matrix_for(data, degradation, layers, experts) for degradation in shown_degradations]
    all_values = finite_values_from(matrices)
    vmax = float(all_values.max()) if all_values.size else 1.0
    vmin = 0.0

    for idx, degradation in enumerate(shown_degradations):
        row_idx, col_idx = divmod(idx, cols)
        ax = axes[row_idx][col_idx]
        matrix = matrices[idx]
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(degradation)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Decoder layer")
        ax.set_xticks(np.arange(len(experts)))
        ax.set_xticklabels([str(expert) for expert in experts])
        ax.set_yticks(np.arange(len(layers)))
        ax.set_yticklabels(layers)

    for idx in range(len(shown_degradations), rows * cols):
        row_idx, col_idx = divmod(idx, cols)
        axes[row_idx][col_idx].axis("off")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(METRICS[metric])
    fig.suptitle(f"Per-degradation expert activation ({metric})", y=1.02)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_expert_curves(data, layers, experts, degradations, metric, output_dir, selected_experts, dpi, image_format):
    if selected_experts is None:
        selected_experts = experts

    output_paths = []
    x = np.arange(len(layers))
    line_degradations = [d for d in degradations if d in data]

    for expert in selected_experts:
        if expert not in experts:
            continue
        fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(layers) + 2.5), 4.6))
        for degradation in line_degradations:
            y_values = []
            for layer in layers:
                y_values.append(data.get(degradation, {}).get(layer, {}).get(expert, np.nan))
            ax.plot(x, y_values, marker="o", linewidth=1.8, label=degradation)

        ax.set_title(f"Expert {expert} activation across decoder layers")
        ax.set_xlabel("Decoder layer")
        ax.set_ylabel(METRICS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        output_path = os.path.join(output_dir, f"expert_{expert}_across_layers.{image_format}")
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_overall_expert_curves(data, layers, experts, metric, output_path, dpi):
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(layers) + 2.5), 4.6))
    for expert in experts:
        y_values = [data.get("ALL", {}).get(layer, {}).get(expert, np.nan) for layer in layers]
        ax.plot(x, y_values, marker="o", linewidth=1.8, label=f"Expert {expert}")

    ax.set_title("All experts across decoder layers (ALL degradations)")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel(METRICS[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_pivot_csv(data, layers, experts, degradations, metric, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["degradation", "layer"] + [f"expert_{expert}" for expert in experts])
        for degradation in degradations:
            for layer in layers:
                row = [degradation, layer]
                for expert in experts:
                    value = data.get(degradation, {}).get(layer, {}).get(expert, "")
                    row.append(value)
                writer.writerow(row)


def parse_experts(value):
    if value is None:
        return None
    experts = []
    for part in value.split(","):
        part = part.strip()
        if part:
            experts.append(int(part))
    return experts


def main():
    parser = argparse.ArgumentParser(description="Visualize MoE expert activation statistics exported by test.py.")
    parser.add_argument("--csv", required=True, help="Path to expert_activation_stats.csv.")
    parser.add_argument("--output_dir", default=None, help="Directory for generated figures. Defaults to <csv_dir>/expert_activation_figures.")
    parser.add_argument("--metric", default="active_fraction", choices=sorted(METRICS), help="CSV metric to visualize.")
    parser.add_argument("--experts", default=None, help="Comma-separated expert ids for per-expert line plots. Defaults to all experts.")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Figure file format.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for raster figures.")
    args = parser.parse_args()

    set_common_style()
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "expert_activation_figures")
    os.makedirs(output_dir, exist_ok=True)

    data, layers, experts, degradations = read_stats(args.csv, args.metric)
    metric_label = METRICS[args.metric]
    selected_experts = parse_experts(args.experts)

    all_matrices = [matrix_for(data, degradation, layers, experts) for degradation in degradations]
    finite_values = finite_values_from(all_matrices)
    vmax = float(finite_values.max()) if finite_values.size else 1.0
    vmin = 0.0

    generated = []
    for degradation in degradations:
        matrix = matrix_for(data, degradation, layers, experts)
        filename = f"{sanitize_name(degradation)}_layer_expert_heatmap.{args.format}"
        output_path = os.path.join(output_dir, filename)
        title = f"{degradation}: expert activation by decoder layer"
        plot_heatmap(matrix, layers, experts, title, metric_label, output_path, vmin, vmax, args.dpi)
        generated.append(output_path)

    combined_path = os.path.join(output_dir, f"per_degradation_layer_expert_heatmaps.{args.format}")
    combined_result = plot_combined_degradation_heatmaps(data, layers, experts, degradations, args.metric, combined_path, args.dpi)
    if combined_result is not None:
        generated.append(combined_result)

    overall_curve_path = os.path.join(output_dir, f"ALL_experts_across_layers.{args.format}")
    plot_overall_expert_curves(data, layers, experts, args.metric, overall_curve_path, args.dpi)
    generated.append(overall_curve_path)

    generated.extend(plot_expert_curves(data, layers, experts, degradations, args.metric, output_dir, selected_experts, args.dpi, args.format))

    pivot_path = os.path.join(output_dir, f"expert_activation_pivot_{args.metric}.csv")
    write_pivot_csv(data, layers, experts, degradations, args.metric, pivot_path)
    generated.append(pivot_path)

    print("Generated files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
