"""
Visualization utilities for optimization results.

Provides publication-quality plots for comparing optimization results.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_optimization_results(
    results: dict,
    output_dir: str | Path = "results",
    show_plots: bool = False,
) -> None:
    """
    Create comprehensive visualization plots comparing optimization results.

    Generates 4 plots:
    1. Accuracy comparison (baseline vs optimized)
    2. F1 Score comparison (macro)
    3. All metrics comparison (grouped bar chart)
    4. Improvement delta (how much optimization helped)

    Args:
        results: Dictionary of optimizer name -> OptimizationResult
        output_dir: Directory to save plots
        show_plots: If True, display plots interactively
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results to plot.")
        return

    # Configure plot style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10

    optimizer_names = list(results.keys())

    # Generate all plots
    _plot_accuracy_comparison(results, optimizer_names, output_dir, show_plots)
    _plot_f1_comparison(results, optimizer_names, output_dir, show_plots)
    _plot_all_metrics(results, optimizer_names, output_dir, show_plots)
    _plot_improvement_delta(results, optimizer_names, output_dir, show_plots)

    print(f"\n✅ All plots saved to {output_dir}/")


def _plot_accuracy_comparison(results, optimizer_names, output_dir, show_plots):
    """Plot accuracy comparison chart."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(optimizer_names))
    width = 0.35

    baseline_accs = [results[name].baseline_accuracy for name in optimizer_names]
    optimized_accs = [results[name].optimized_accuracy for name in optimizer_names]

    ax.bar(x - width / 2, baseline_accs, width, label="Baseline", color="#3498db")
    ax.bar(x + width / 2, optimized_accs, width, label="Optimized", color="#2ecc71")

    ax.set_xlabel("Optimizer", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Classification Accuracy: Baseline vs Optimized", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()  # type: ignore[attr-defined]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,  # type: ignore[attr-defined]
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")

    if show_plots:
        plt.show()
    plt.close()


def _plot_f1_comparison(results, optimizer_names, output_dir, show_plots):
    """Plot F1 score comparison chart."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(optimizer_names))
    width = 0.35

    baseline_f1s = [results[name].baseline_f1_macro or 0 for name in optimizer_names]
    optimized_f1s = [results[name].optimized_f1_macro or 0 for name in optimizer_names]

    ax.bar(x - width / 2, baseline_f1s, width, label="Baseline", color="#e74c3c")
    ax.bar(x + width / 2, optimized_f1s, width, label="Optimized", color="#f39c12")

    ax.set_xlabel("Optimizer", fontweight="bold")
    ax.set_ylabel("F1 Score (Macro)", fontweight="bold")
    ax.set_title("F1 Score (Macro): Baseline vs Optimized", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()  # type: ignore[attr-defined]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,  # type: ignore[attr-defined]
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'f1_comparison.png'}")

    if show_plots:
        plt.show()
    plt.close()


def _plot_all_metrics(results, optimizer_names, output_dir, show_plots):
    """Plot all metrics comparison (grouped bar chart)."""
    _fig, ax = plt.subplots(figsize=(14, 8))

    metrics_config = [
        ("Accuracy (%)", "baseline_accuracy", "optimized_accuracy", 1.0),
        ("F1 (Macro)", "baseline_f1_macro", "optimized_f1_macro", 100.0),
        ("Precision (Macro)", "baseline_precision_macro", "optimized_precision_macro", 100.0),
        ("Recall (Macro)", "baseline_recall_macro", "optimized_recall_macro", 100.0),
    ]

    n_metrics = len(metrics_config)
    n_optimizers = len(optimizer_names)
    x = np.arange(n_metrics)
    width = 0.8 / (n_optimizers * 2)

    colors_baseline = ["#3498db", "#e74c3c", "#9b59b6", "#1abc9c"]
    colors_optimized = ["#2ecc71", "#f39c12", "#8e44ad", "#16a085"]

    for i, opt_name in enumerate(optimizer_names):
        result = results[opt_name]
        baseline_vals = []
        optimized_vals = []

        for _, baseline_attr, optimized_attr, scale in metrics_config:
            baseline_val = getattr(result, baseline_attr, 0) or 0
            optimized_val = getattr(result, optimized_attr, 0) or 0

            # Scale non-percentage metrics to 0-100 range
            if "accuracy" not in baseline_attr.lower():
                baseline_val *= scale
                optimized_val *= scale

            baseline_vals.append(baseline_val)
            optimized_vals.append(optimized_val)

        offset = (i * 2 - n_optimizers + 1) * width
        ax.bar(
            x + offset,
            baseline_vals,
            width,
            label=f"{opt_name} (Baseline)",
            color=colors_baseline[i % len(colors_baseline)],
            alpha=0.7,
        )
        ax.bar(
            x + offset + width,
            optimized_vals,
            width,
            label=f"{opt_name} (Optimized)",
            color=colors_optimized[i % len(colors_optimized)],
            alpha=0.9,
        )

    ax.set_xlabel("Metrics", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("All Classification Metrics: Baseline vs Optimized", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _, _ in metrics_config])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'all_metrics_comparison.png'}")

    if show_plots:
        plt.show()
    plt.close()


def _plot_improvement_delta(results, optimizer_names, output_dir, show_plots):
    """Plot improvement delta chart."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    improvements = [results[name].improvement for name in optimizer_names]
    colors = ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements]

    bars = ax.bar(optimizer_names, improvements, color=colors, alpha=0.7)

    ax.set_xlabel("Optimizer", fontweight="bold")
    ax.set_ylabel("Accuracy Improvement (%)", fontweight="bold")
    ax.set_title("Optimization Improvement by Optimizer", fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()  # type: ignore[attr-defined]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,  # type: ignore[attr-defined]
            height,
            f"{height:+.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_delta.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'improvement_delta.png'}")

    if show_plots:
        plt.show()
    plt.close()


def plot_single_optimizer_results(
    result,
    output_path: str | Path,
    show_plot: bool = False,
) -> None:
    """
    Create a simple 2-panel comparison plot for a single optimizer.

    Args:
        result: The optimization result to plot
        output_path: Path to save the plot
        show_plot: If True, display plot interactively
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Accuracy comparison
    categories = ["Baseline", "Optimized"]
    accuracies = [result.baseline_accuracy, result.optimized_accuracy]
    colors = ["#3498db", "#2ecc71"]

    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title(f"{result.optimizer_name} - Accuracy", fontweight="bold")
    ax1.set_ylim([0, 100])
    ax1.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()  # type: ignore[attr-defined]
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,  # type: ignore[attr-defined]
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Panel 2: All metrics
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    baseline_vals = [
        result.baseline_accuracy,
        (result.baseline_f1_macro or 0) * 100,
        (result.baseline_precision_macro or 0) * 100,
        (result.baseline_recall_macro or 0) * 100,
    ]
    optimized_vals = [
        result.optimized_accuracy,
        (result.optimized_f1_macro or 0) * 100,
        (result.optimized_precision_macro or 0) * 100,
        (result.optimized_recall_macro or 0) * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax2.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#3498db")
    ax2.bar(x + width / 2, optimized_vals, width, label="Optimized", color="#2ecc71")

    ax2.set_ylabel("Score (%)", fontweight="bold")
    ax2.set_title(f"{result.optimizer_name} - All Metrics", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if show_plot:
        plt.show()
    plt.close()
