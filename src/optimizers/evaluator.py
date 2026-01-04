"""
Evaluation utilities for DSPy classifiers.

Provides metric computation and evaluation helpers.
"""

import dspy
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(
    testset: list[dspy.Example],
    classifier: dspy.Predict,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        testset: Test examples
        classifier: The classifier to evaluate

    Returns:
        Dictionary with accuracy, precision, recall, F1 scores (macro & weighted)
    """
    y_true = []
    y_pred = []

    # Collect predictions
    for ex in testset:
        prediction = classifier(query=ex.query)
        y_true.append(ex.intent.lower().strip())
        y_pred.append(prediction.intent.lower().strip())

    # Compute all metrics
    # Note: type: ignore comments are due to sklearn type stubs being outdated
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0.0  # type: ignore[arg-type]
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0.0  # type: ignore[arg-type]
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0.0  # type: ignore[arg-type]
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0.0  # type: ignore[arg-type]
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0.0  # type: ignore[arg-type]
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0.0  # type: ignore[arg-type]
        ),
    }


def print_metrics_summary(metrics: dict, label: str = "Metrics") -> None:
    """
    Print a formatted summary of classification metrics.

    Args:
        metrics: Dictionary of metrics from compute_classification_metrics
        label: Label for the summary (e.g., "Baseline", "Optimized")
    """
    print(f"✓ {label} accuracy: {metrics['accuracy']:.1f}%")
    print(f"✓ {label} F1 (macro): {metrics['f1_macro']:.3f}")
    print(f"✓ {label} precision (macro): {metrics['precision_macro']:.3f}")
    print(f"✓ {label} recall (macro): {metrics['recall_macro']:.3f}")


def print_detailed_comparison(
    baseline_metrics: dict,
    optimized_metrics: dict,
) -> None:
    """
    Print a detailed comparison table of baseline vs optimized metrics.

    Args:
        baseline_metrics: Metrics from baseline classifier
        optimized_metrics: Metrics from optimized classifier
    """
    print("\n" + "=" * 70)
    print("DETAILED METRICS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Baseline':>15} {'Optimized':>15} {'Change':>8}")
    print("-" * 70)

    # Accuracy
    print(
        f"{'Accuracy':<30} {baseline_metrics['accuracy']:>14.1f}% "
        f"{optimized_metrics['accuracy']:>14.1f}% "
        f"{optimized_metrics['accuracy'] - baseline_metrics['accuracy']:>+7.1f}%"
    )

    # F1, Precision, Recall (both macro and weighted)
    for metric_name in ["f1", "precision", "recall"]:
        for avg_type in ["macro", "weighted"]:
            key = f"{metric_name}_{avg_type}"
            baseline_val = baseline_metrics[key]
            optimized_val = optimized_metrics[key]
            display_name = f"{metric_name.capitalize()} ({avg_type})"

            print(
                f"{display_name:<30} {baseline_val:>15.3f} "
                f"{optimized_val:>15.3f} "
                f"{optimized_val - baseline_val:>+8.3f}"
            )

    print("=" * 70)
