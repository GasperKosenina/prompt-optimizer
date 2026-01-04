"""
Optimizer Runner Module

This module provides utilities for running and comparing DSPy optimizers.
It supports:
- BootstrapFewShot: Simple, fast, low-cost optimizer
- MIPROv2: Advanced optimizer with better results but higher cost

The runner tracks metrics, costs, and saves results for comparison.
"""

import json
import time
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@dataclass
class OptimizationResult:
    """Results from an optimization run."""

    optimizer_name: str
    baseline_accuracy: float
    optimized_accuracy: float
    improvement: float
    train_size: int
    test_size: int
    duration_seconds: float
    timestamp: str

    # Optimizer-specific settings
    settings: dict

    # Additional classification metrics (baseline)
    baseline_precision_macro: float | None = None
    baseline_recall_macro: float | None = None
    baseline_f1_macro: float | None = None
    baseline_precision_weighted: float | None = None
    baseline_recall_weighted: float | None = None
    baseline_f1_weighted: float | None = None

    # Additional classification metrics (optimized)
    optimized_precision_macro: float | None = None
    optimized_recall_macro: float | None = None
    optimized_f1_macro: float | None = None
    optimized_precision_weighted: float | None = None
    optimized_recall_weighted: float | None = None
    optimized_f1_weighted: float | None = None

    # Cost tracking (optional, added if available)
    total_api_calls: int | None = None
    estimated_cost_usd: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def __str__(self) -> str:
        cost_str = (
            f" [${self.estimated_cost_usd:.2f}]" if self.estimated_cost_usd else ""
        )
        return (
            f"{self.optimizer_name}: "
            f"Acc: {self.baseline_accuracy:.1f}% → {self.optimized_accuracy:.1f}% "
            f"({self.improvement:+.1f}%), "
            f"F1: {self.baseline_f1_macro:.3f} → {self.optimized_f1_macro:.3f}"
            f"{cost_str}"
        )


def _print_progress(message: str, elapsed_seconds: float | None = None):
    """Print a progress message with optional elapsed time."""
    if elapsed_seconds is not None:
        elapsed_str = f"[{elapsed_seconds:.1f}s elapsed]"
        print(f"{message} {elapsed_str}")
    else:
        print(message)
    sys.stdout.flush()


def _compute_classification_metrics(
    testset: list[dspy.Example],
    classifier: dspy.Predict,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        testset: Test examples
        classifier: The classifier to evaluate

    Returns:
        Dictionary with accuracy, precision, recall, F1 scores
    """
    y_true = []
    y_pred = []

    # Collect predictions
    for ex in testset:
        prediction = classifier(query=ex.query)
        y_true.append(ex.intent.lower().strip())
        y_pred.append(prediction.intent.lower().strip())

    # Compute metrics
    # Note: zero_division=0.0 handles cases where a class has no predictions
    # type: ignore comments are needed due to sklearn type stubs being outdated
    metrics = {
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

    return metrics


def _check_mipro_dataset_size(trainset_size: int, auto: str) -> bool:
    """
    Check if dataset size is reasonable for MIPROv2 and warn user.

    Returns True if user confirms to continue, False otherwise.
    """
    warnings = []

    if auto == "light" and trainset_size > 100:
        warnings.append(
            f"⚠️  Large dataset ({trainset_size} examples) for MIPROv2 light mode"
        )
        warnings.append("   Recommended: 50-100 examples for light mode")
        warnings.append("   This may take 10-30 minutes and cost $1-5")
    elif auto == "medium" and trainset_size > 300:
        warnings.append(
            f"⚠️  Large dataset ({trainset_size} examples) for MIPROv2 medium mode"
        )
        warnings.append("   Recommended: 200-300 examples for medium mode")
        warnings.append("   This may take 30-90 minutes and cost $5-20")
    elif auto == "heavy" and trainset_size > 100:
        warnings.append(
            f"⚠️  Large dataset ({trainset_size} examples) for MIPROv2 heavy mode"
        )
        warnings.append("   Recommended: <100 examples for heavy mode")
        warnings.append("   This may take HOURS and cost $20-100+")

    if warnings:
        print("\n" + "!" * 60)
        for warning in warnings:
            print(warning)
        print("!" * 60)

        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        return response in ["y", "yes"]

    return True


def run_bootstrap_fewshot(
    trainset: list[dspy.Example],
    testset: list[dspy.Example],
    classifier_class: type,
    metric_fn: Callable,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    max_rounds: int = 1,
    verbose: bool = True,
) -> tuple[dspy.Predict, OptimizationResult]:
    """
    Run BootstrapFewShot optimization.

    BootstrapFewShot works by:
    1. Running the model on training examples
    2. Selecting examples where the model got the answer correct
    3. Using these as few-shot demonstrations in the prompt

    Args:
        trainset: Training examples (DSPy Examples)
        testset: Test examples for evaluation
        classifier_class: The DSPy Signature class (e.g., IntentClassifier)
        metric_fn: Function(example, prediction) -> bool
        max_bootstrapped_demos: Max auto-generated demos
        max_labeled_demos: Max labeled examples from trainset
        max_rounds: Number of optimization rounds
        verbose: Print progress

    Returns:
        Tuple of (optimized_classifier, OptimizationResult)

    Example:
        from src.modules.intent_classifier import IntentClassifier, accuracy_metric
        from src.data.loader import load_intent_classification_data

        trainset, testset = load_intent_classification_data(n_train=200, n_test=100)
        optimized, result = run_bootstrap_fewshot(
            trainset, testset, IntentClassifier, accuracy_metric
        )
        print(result)
    """
    if verbose:
        print("=" * 60)
        print("Running BootstrapFewShot Optimization")
        print("=" * 60)
        print(f"Train size: {len(trainset)}, Test size: {len(testset)}")
        print(
            f"Settings: max_bootstrapped_demos={max_bootstrapped_demos}, "
            f"max_labeled_demos={max_labeled_demos}, max_rounds={max_rounds}"
        )

    start_time = time.time()

    # Create baseline classifier
    baseline = dspy.Predict(classifier_class)

    # Evaluate baseline
    if verbose:
        _print_progress("\n[1/4] Evaluating baseline classifier...")
    baseline_metrics = _compute_classification_metrics(testset, baseline)
    baseline_accuracy = baseline_metrics["accuracy"]

    if verbose:
        print(f"       ✓ Baseline accuracy: {baseline_accuracy:.1f}%")
        print(f"       ✓ Baseline F1 (macro): {baseline_metrics['f1_macro']:.3f}")
        print(
            f"       ✓ Baseline precision (macro): {baseline_metrics['precision_macro']:.3f}"
        )
        print(f"       ✓ Baseline recall (macro): {baseline_metrics['recall_macro']:.3f}")

    # Create and run optimizer
    if verbose:
        _print_progress("\n[2/4] Running BootstrapFewShot optimization...")
        print(
            f"       • Bootstrapping few-shot examples from {len(trainset)} training examples"
        )
        print(
            f"       • This will take ~{len(trainset) * 0.5:.0f}-{len(trainset) * 1:.0f} seconds"
        )

    optimize_start = time.time()
    optimizer = BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
    )

    optimized = optimizer.compile(
        student=dspy.Predict(classifier_class),
        trainset=trainset,
    )
    optimize_duration = time.time() - optimize_start

    if verbose:
        print(f"       ✓ Optimization complete in {optimize_duration:.1f}s")

    # Evaluate optimized classifier
    if verbose:
        _print_progress("\n[3/4] Evaluating optimized classifier...")
    optimized_metrics = _compute_classification_metrics(testset, optimized)
    optimized_accuracy = optimized_metrics["accuracy"]

    duration = time.time() - start_time

    if verbose:
        print(f"       ✓ Optimized accuracy: {optimized_accuracy:.1f}%")
        print(f"       ✓ Optimized F1 (macro): {optimized_metrics['f1_macro']:.3f}")
        print(
            f"       ✓ Optimized precision (macro): {optimized_metrics['precision_macro']:.3f}"
        )
        print(f"       ✓ Optimized recall (macro): {optimized_metrics['recall_macro']:.3f}")
        print(f"\n       Improvement: {optimized_accuracy - baseline_accuracy:+.1f}%")
        print(f"       Duration: {duration:.1f}s")

        # Print detailed metrics table
        print("\n" + "=" * 70)
        print("DETAILED METRICS COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<30} {'Baseline':>15} {'Optimized':>15} {'Change':>8}")
        print("-" * 70)
        print(
            f"{'Accuracy':<30} {baseline_accuracy:>14.1f}% {optimized_accuracy:>14.1f}% "
            f"{optimized_accuracy - baseline_accuracy:>+7.1f}%"
        )
        print(
            f"{'F1 Score (macro)':<30} {baseline_metrics['f1_macro']:>15.3f} "
            f"{optimized_metrics['f1_macro']:>15.3f} "
            f"{optimized_metrics['f1_macro'] - baseline_metrics['f1_macro']:>+8.3f}"
        )
        print(
            f"{'Precision (macro)':<30} {baseline_metrics['precision_macro']:>15.3f} "
            f"{optimized_metrics['precision_macro']:>15.3f} "
            f"{optimized_metrics['precision_macro'] - baseline_metrics['precision_macro']:>+8.3f}"
        )
        print(
            f"{'Recall (macro)':<30} {baseline_metrics['recall_macro']:>15.3f} "
            f"{optimized_metrics['recall_macro']:>15.3f} "
            f"{optimized_metrics['recall_macro'] - baseline_metrics['recall_macro']:>+8.3f}"
        )
        print(
            f"{'F1 Score (weighted)':<30} {baseline_metrics['f1_weighted']:>15.3f} "
            f"{optimized_metrics['f1_weighted']:>15.3f} "
            f"{optimized_metrics['f1_weighted'] - baseline_metrics['f1_weighted']:>+8.3f}"
        )
        print(
            f"{'Precision (weighted)':<30} {baseline_metrics['precision_weighted']:>15.3f} "
            f"{optimized_metrics['precision_weighted']:>15.3f} "
            f"{optimized_metrics['precision_weighted'] - baseline_metrics['precision_weighted']:>+8.3f}"
        )
        print(
            f"{'Recall (weighted)':<30} {baseline_metrics['recall_weighted']:>15.3f} "
            f"{optimized_metrics['recall_weighted']:>15.3f} "
            f"{optimized_metrics['recall_weighted'] - baseline_metrics['recall_weighted']:>+8.3f}"
        )
        print("=" * 70)

    result = OptimizationResult(
        optimizer_name="BootstrapFewShot",
        baseline_accuracy=baseline_accuracy,
        optimized_accuracy=optimized_accuracy,
        improvement=optimized_accuracy - baseline_accuracy,
        train_size=len(trainset),
        test_size=len(testset),
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
        settings={
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "max_rounds": max_rounds,
        },
        # Baseline metrics
        baseline_precision_macro=baseline_metrics["precision_macro"],
        baseline_recall_macro=baseline_metrics["recall_macro"],
        baseline_f1_macro=baseline_metrics["f1_macro"],
        baseline_precision_weighted=baseline_metrics["precision_weighted"],
        baseline_recall_weighted=baseline_metrics["recall_weighted"],
        baseline_f1_weighted=baseline_metrics["f1_weighted"],
        # Optimized metrics
        optimized_precision_macro=optimized_metrics["precision_macro"],
        optimized_recall_macro=optimized_metrics["recall_macro"],
        optimized_f1_macro=optimized_metrics["f1_macro"],
        optimized_precision_weighted=optimized_metrics["precision_weighted"],
        optimized_recall_weighted=optimized_metrics["recall_weighted"],
        optimized_f1_weighted=optimized_metrics["f1_weighted"],
    )

    return optimized, result


def run_mipro_v2(
    trainset: list[dspy.Example],
    testset: list[dspy.Example],
    classifier_class: type,
    metric_fn: Callable,
    auto: Literal["light", "medium", "heavy"] = "light",
    num_threads: int = 4,
    verbose: bool = True,
) -> tuple[dspy.Predict, OptimizationResult]:
    """
    Run MIPROv2 optimization.

    MIPROv2 (Multi-prompt Instruction Proposal Optimizer) works by:
    1. Generating candidate instructions using a meta-prompt
    2. Evaluating each candidate on a subset of training data
    3. Using Bayesian optimization to find the best instruction

    This is more expensive but often produces better results.

    Args:
        trainset: Training examples (DSPy Examples)
        testset: Test examples for evaluation
        classifier_class: The DSPy Signature class
        metric_fn: Function(example, prediction) -> bool
        auto: Optimization intensity ("light", "medium", "heavy")
              - light: Fast, low cost (~5-10 trials)
              - medium: Balanced (~20-30 trials)
              - heavy: Thorough, high cost (~50+ trials)
        num_threads: Parallel evaluation threads
        verbose: Print progress

    Returns:
        Tuple of (optimized_classifier, OptimizationResult)

    Example:
        optimized, result = run_mipro_v2(
            trainset, testset, IntentClassifier, accuracy_metric, auto="light"
        )
    """
    if verbose:
        print("=" * 60)
        print("Running MIPROv2 Optimization")
        print("=" * 60)
        print(f"Train size: {len(trainset)}, Test size: {len(testset)}")
        print(f"Settings: auto={auto}, num_threads={num_threads}")
        print("\nNote: MIPROv2 may take several minutes and use more API calls.")

    # Check dataset size and get user confirmation if needed
    if not _check_mipro_dataset_size(len(trainset), auto):
        print("\n❌ Optimization cancelled by user.")
        # Return a baseline result indicating cancellation
        baseline = dspy.Predict(classifier_class)
        baseline_metrics = _compute_classification_metrics(testset, baseline)
        baseline_accuracy = baseline_metrics["accuracy"]

        result = OptimizationResult(
            optimizer_name="MIPROv2",
            baseline_accuracy=baseline_accuracy,
            optimized_accuracy=baseline_accuracy,
            improvement=0.0,
            train_size=len(trainset),
            test_size=len(testset),
            duration_seconds=0.0,
            timestamp=datetime.now().isoformat(),
            settings={
                "auto": auto,
                "num_threads": num_threads,
                "cancelled": True,
            },
            # Set baseline and optimized to same values since cancelled
            baseline_precision_macro=baseline_metrics["precision_macro"],
            baseline_recall_macro=baseline_metrics["recall_macro"],
            baseline_f1_macro=baseline_metrics["f1_macro"],
            baseline_precision_weighted=baseline_metrics["precision_weighted"],
            baseline_recall_weighted=baseline_metrics["recall_weighted"],
            baseline_f1_weighted=baseline_metrics["f1_weighted"],
            optimized_precision_macro=baseline_metrics["precision_macro"],
            optimized_recall_macro=baseline_metrics["recall_macro"],
            optimized_f1_macro=baseline_metrics["f1_macro"],
            optimized_precision_weighted=baseline_metrics["precision_weighted"],
            optimized_recall_weighted=baseline_metrics["recall_weighted"],
            optimized_f1_weighted=baseline_metrics["f1_weighted"],
        )
        return baseline, result

    start_time = time.time()

    # Create baseline classifier
    baseline = dspy.Predict(classifier_class)

    # Evaluate baseline
    if verbose:
        print("\n[1/3] Evaluating baseline...")
    baseline_metrics = _compute_classification_metrics(testset, baseline)
    baseline_accuracy = baseline_metrics["accuracy"]

    if verbose:
        print(f"         ✓ Baseline accuracy: {baseline_accuracy:.1f}%")
        print(f"         ✓ Baseline F1 (macro): {baseline_metrics['f1_macro']:.3f}")
        print(
            f"         ✓ Baseline precision (macro): {baseline_metrics['precision_macro']:.3f}"
        )
        print(f"         ✓ Baseline recall (macro): {baseline_metrics['recall_macro']:.3f}")

    # Create and run optimizer
    if verbose:
        print("\n[2/3] Optimizing (this may take a while)...")

    optimizer = MIPROv2(
        metric=metric_fn,
        auto=auto,
        num_threads=num_threads,
    )

    optimized = optimizer.compile(
        student=dspy.Predict(classifier_class),
        trainset=trainset,
    )

    # Evaluate optimized classifier
    if verbose:
        print("\n[3/3] Evaluating optimized classifier...")
    optimized_metrics = _compute_classification_metrics(testset, optimized)
    optimized_accuracy = optimized_metrics["accuracy"]

    duration = time.time() - start_time

    if verbose:
        print(f"         ✓ Optimized accuracy: {optimized_accuracy:.1f}%")
        print(f"         ✓ Optimized F1 (macro): {optimized_metrics['f1_macro']:.3f}")
        print(
            f"         ✓ Optimized precision (macro): {optimized_metrics['precision_macro']:.3f}"
        )
        print(f"         ✓ Optimized recall (macro): {optimized_metrics['recall_macro']:.3f}")
        print(f"\n         Improvement: {optimized_accuracy - baseline_accuracy:+.1f}%")
        print(f"         Duration: {duration:.1f}s")

        # Print detailed metrics table
        print("\n" + "=" * 70)
        print("DETAILED METRICS COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<30} {'Baseline':>15} {'Optimized':>15} {'Change':>8}")
        print("-" * 70)
        print(
            f"{'Accuracy':<30} {baseline_accuracy:>14.1f}% {optimized_accuracy:>14.1f}% "
            f"{optimized_accuracy - baseline_accuracy:>+7.1f}%"
        )
        print(
            f"{'F1 Score (macro)':<30} {baseline_metrics['f1_macro']:>15.3f} "
            f"{optimized_metrics['f1_macro']:>15.3f} "
            f"{optimized_metrics['f1_macro'] - baseline_metrics['f1_macro']:>+8.3f}"
        )
        print(
            f"{'Precision (macro)':<30} {baseline_metrics['precision_macro']:>15.3f} "
            f"{optimized_metrics['precision_macro']:>15.3f} "
            f"{optimized_metrics['precision_macro'] - baseline_metrics['precision_macro']:>+8.3f}"
        )
        print(
            f"{'Recall (macro)':<30} {baseline_metrics['recall_macro']:>15.3f} "
            f"{optimized_metrics['recall_macro']:>15.3f} "
            f"{optimized_metrics['recall_macro'] - baseline_metrics['recall_macro']:>+8.3f}"
        )
        print(
            f"{'F1 Score (weighted)':<30} {baseline_metrics['f1_weighted']:>15.3f} "
            f"{optimized_metrics['f1_weighted']:>15.3f} "
            f"{optimized_metrics['f1_weighted'] - baseline_metrics['f1_weighted']:>+8.3f}"
        )
        print(
            f"{'Precision (weighted)':<30} {baseline_metrics['precision_weighted']:>15.3f} "
            f"{optimized_metrics['precision_weighted']:>15.3f} "
            f"{optimized_metrics['precision_weighted'] - baseline_metrics['precision_weighted']:>+8.3f}"
        )
        print(
            f"{'Recall (weighted)':<30} {baseline_metrics['recall_weighted']:>15.3f} "
            f"{optimized_metrics['recall_weighted']:>15.3f} "
            f"{optimized_metrics['recall_weighted'] - baseline_metrics['recall_weighted']:>+8.3f}"
        )
        print("=" * 70)

    result = OptimizationResult(
        optimizer_name="MIPROv2",
        baseline_accuracy=baseline_accuracy,
        optimized_accuracy=optimized_accuracy,
        improvement=optimized_accuracy - baseline_accuracy,
        train_size=len(trainset),
        test_size=len(testset),
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
        settings={
            "auto": auto,
            "num_threads": num_threads,
        },
        # Baseline metrics
        baseline_precision_macro=baseline_metrics["precision_macro"],
        baseline_recall_macro=baseline_metrics["recall_macro"],
        baseline_f1_macro=baseline_metrics["f1_macro"],
        baseline_precision_weighted=baseline_metrics["precision_weighted"],
        baseline_recall_weighted=baseline_metrics["recall_weighted"],
        baseline_f1_weighted=baseline_metrics["f1_weighted"],
        # Optimized metrics
        optimized_precision_macro=optimized_metrics["precision_macro"],
        optimized_recall_macro=optimized_metrics["recall_macro"],
        optimized_f1_macro=optimized_metrics["f1_macro"],
        optimized_precision_weighted=optimized_metrics["precision_weighted"],
        optimized_recall_weighted=optimized_metrics["recall_weighted"],
        optimized_f1_weighted=optimized_metrics["f1_weighted"],
    )

    return optimized, result


def compare_optimizers(
    trainset: list[dspy.Example],
    testset: list[dspy.Example],
    classifier_class: type,
    metric_fn: Callable,
    optimizers: list[Literal["bootstrap", "mipro_light", "mipro_medium"]] | None = None,
    verbose: bool = True,
) -> dict[str, OptimizationResult]:
    """
    Run multiple optimizers and compare results.

    Args:
        trainset: Training examples
        testset: Test examples
        classifier_class: The DSPy Signature class
        metric_fn: Metric function
        optimizers: List of optimizers to run. Options:
            - "bootstrap": BootstrapFewShot
            - "mipro_light": MIPROv2 with auto="light"
            - "mipro_medium": MIPROv2 with auto="medium"
        verbose: Print progress

    Returns:
        Dictionary mapping optimizer name to OptimizationResult
    """
    if optimizers is None:
        optimizers = ["bootstrap", "mipro_light"]

    results = {}

    for opt_name in optimizers:
        if verbose:
            print(f"\n{'#' * 60}")
            print(f"# Running: {opt_name}")
            print(f"{'#' * 60}\n")

        if opt_name == "bootstrap":
            _, result = run_bootstrap_fewshot(
                trainset, testset, classifier_class, metric_fn, verbose=verbose
            )
        elif opt_name == "mipro_light":
            _, result = run_mipro_v2(
                trainset,
                testset,
                classifier_class,
                metric_fn,
                auto="light",
                verbose=verbose,
            )
        else:  # mipro_medium
            _, result = run_mipro_v2(
                trainset,
                testset,
                classifier_class,
                metric_fn,
                auto="medium",
                verbose=verbose,
            )

        results[opt_name] = result

    # Print comparison summary
    if verbose:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY - ACCURACY")
        print("=" * 80)
        print(
            f"{'Optimizer':<20} {'Baseline':>10} {'Optimized':>10} {'Improvement':>12} {'Time':>10}"
        )
        print("-" * 80)
        for name, res in results.items():
            print(
                f"{name:<20} {res.baseline_accuracy:>9.1f}% {res.optimized_accuracy:>9.1f}% "
                f"{res.improvement:>+11.1f}% {res.duration_seconds:>9.1f}s"
            )

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY - F1 SCORES (MACRO)")
        print("=" * 80)
        print(
            f"{'Optimizer':<20} {'Baseline F1':>12} {'Optimized F1':>13} {'Improvement':>12}"
        )
        print("-" * 80)
        for name, res in results.items():
            baseline_f1 = res.baseline_f1_macro or 0
            optimized_f1 = res.optimized_f1_macro or 0
            improvement = optimized_f1 - baseline_f1
            print(
                f"{name:<20} {baseline_f1:>12.3f} {optimized_f1:>13.3f} "
                f"{improvement:>+12.3f}"
            )

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY - PRECISION & RECALL (MACRO)")
        print("=" * 80)
        print(
            f"{'Optimizer':<20} {'Base P':>8} {'Opt P':>8} {'Base R':>8} {'Opt R':>8}"
        )
        print("-" * 80)
        for name, res in results.items():
            base_p = res.baseline_precision_macro or 0
            opt_p = res.optimized_precision_macro or 0
            base_r = res.baseline_recall_macro or 0
            opt_r = res.optimized_recall_macro or 0
            print(
                f"{name:<20} {base_p:>8.3f} {opt_p:>8.3f} {base_r:>8.3f} {opt_r:>8.3f}"
            )
        print("=" * 80)

    return results


def save_results(
    results: dict[str, OptimizationResult],
    filepath: str | Path,
) -> None:
    """
    Save optimization results to a JSON file.

    Args:
        results: Dictionary of optimizer name -> OptimizationResult
        filepath: Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {name: res.to_dict() for name, res in results.items()}

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {filepath}")


def load_results(filepath: str | Path) -> dict[str, OptimizationResult]:
    """
    Load optimization results from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary of optimizer name -> OptimizationResult
    """
    with open(filepath) as f:
        data = json.load(f)

    return {name: OptimizationResult(**res_dict) for name, res_dict in data.items()}


def plot_optimization_results(
    results: dict[str, OptimizationResult],
    output_dir: str | Path = "results",
    show_plots: bool = False,
) -> None:
    """
    Create visualization plots comparing optimization results.

    Generates multiple plots:
    1. Accuracy comparison (baseline vs optimized)
    2. F1 Score comparison (macro)
    3. All metrics comparison (detailed)

    Args:
        results: Dictionary of optimizer name -> OptimizationResult
        output_dir: Directory to save plots
        show_plots: If True, display plots interactively (default: False, just save)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style for publication-quality plots
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10

    # Extract data for plotting
    optimizer_names = list(results.keys())
    if not optimizer_names:
        print("No results to plot.")
        return

    # ============================================================
    # Plot 1: Accuracy Comparison
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(optimizer_names))
    width = 0.35

    baseline_accs = [results[name].baseline_accuracy for name in optimizer_names]
    optimized_accs = [results[name].optimized_accuracy for name in optimizer_names]

    bars1 = ax.bar(x - width / 2, baseline_accs, width, label="Baseline", color="#3498db")
    bars2 = ax.bar(
        x + width / 2, optimized_accs, width, label="Optimized", color="#2ecc71"
    )

    ax.set_xlabel("Optimizer", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Classification Accuracy: Baseline vs Optimized", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
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

    # ============================================================
    # Plot 2: F1 Score Comparison (Macro)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_f1s = [
        results[name].baseline_f1_macro or 0 for name in optimizer_names
    ]
    optimized_f1s = [
        results[name].optimized_f1_macro or 0 for name in optimizer_names
    ]

    bars1 = ax.bar(x - width / 2, baseline_f1s, width, label="Baseline", color="#e74c3c")
    bars2 = ax.bar(
        x + width / 2, optimized_f1s, width, label="Optimized", color="#f39c12"
    )

    ax.set_xlabel("Optimizer", fontweight="bold")
    ax.set_ylabel("F1 Score (Macro)", fontweight="bold")
    ax.set_title("F1 Score (Macro): Baseline vs Optimized", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
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

    # ============================================================
    # Plot 3: All Metrics Comparison (Grouped Bar Chart)
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics_to_plot = [
        ("Accuracy (%)", "baseline_accuracy", "optimized_accuracy", 1.0),
        ("F1 (Macro)", "baseline_f1_macro", "optimized_f1_macro", 100.0),
        ("Precision (Macro)", "baseline_precision_macro", "optimized_precision_macro", 100.0),
        ("Recall (Macro)", "baseline_recall_macro", "optimized_recall_macro", 100.0),
    ]

    n_metrics = len(metrics_to_plot)
    n_optimizers = len(optimizer_names)
    x = np.arange(n_metrics)
    width = 0.8 / (n_optimizers * 2)  # Dynamic width based on number of optimizers

    colors_baseline = ["#3498db", "#e74c3c", "#9b59b6", "#1abc9c"]
    colors_optimized = ["#2ecc71", "#f39c12", "#8e44ad", "#16a085"]

    for i, opt_name in enumerate(optimizer_names):
        result = results[opt_name]
        baseline_values = []
        optimized_values = []

        for _, baseline_attr, optimized_attr, scale in metrics_to_plot:
            baseline_val = getattr(result, baseline_attr, 0) or 0
            optimized_val = getattr(result, optimized_attr, 0) or 0

            # Scale percentage metrics, leave ratios as-is
            if "accuracy" in baseline_attr.lower():
                baseline_values.append(baseline_val)
                optimized_values.append(optimized_val)
            else:
                baseline_values.append(baseline_val * scale)
                optimized_values.append(optimized_val * scale)

        offset = (i * 2 - n_optimizers + 1) * width
        ax.bar(
            x + offset,
            baseline_values,
            width,
            label=f"{opt_name} (Baseline)",
            color=colors_baseline[i % len(colors_baseline)],
            alpha=0.7,
        )
        ax.bar(
            x + offset + width,
            optimized_values,
            width,
            label=f"{opt_name} (Optimized)",
            color=colors_optimized[i % len(colors_optimized)],
            alpha=0.9,
        )

    ax.set_xlabel("Metrics", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("All Classification Metrics: Baseline vs Optimized", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _, _ in metrics_to_plot])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "all_metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'all_metrics_comparison.png'}")
    if show_plots:
        plt.show()
    plt.close()

    # ============================================================
    # Plot 4: Improvement Delta (How much did optimization help?)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    improvements = [results[name].improvement for name in optimizer_names]
    colors = ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements]

    bars = ax.bar(optimizer_names, improvements, color=colors, alpha=0.7)

    ax.set_xlabel("Optimizer", fontweight="bold")
    ax.set_ylabel("Accuracy Improvement (%)", fontweight="bold")
    ax.set_title("Optimization Improvement by Optimizer", fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
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

    print(f"\n✅ All plots saved to {output_dir}/")


def plot_single_optimizer_results(
    result: OptimizationResult,
    output_path: str | Path,
    show_plot: bool = False,
) -> None:
    """
    Create a simple comparison plot for a single optimizer result.

    Args:
        result: The optimization result to plot
        output_path: Path to save the plot
        show_plot: If True, display plot interactively
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Accuracy
    categories = ["Baseline", "Optimized"]
    accuracies = [result.baseline_accuracy, result.optimized_accuracy]
    colors = ["#3498db", "#2ecc71"]

    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title(f"{result.optimizer_name} - Accuracy", fontweight="bold")
    ax1.set_ylim([0, 100])
    ax1.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Plot 2: All Metrics
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


# CLI entry point
if __name__ == "__main__":
    import argparse
    import sys

    from dotenv import load_dotenv

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_intent_classification_data  # type: ignore[import-not-found]
    from src.modules.intent_classifier import (  # type: ignore[import-not-found]
        IntentClassifier,
        accuracy_metric,
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run DSPy optimizers on intent classification task"
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        choices=["bootstrap", "mipro", "both"],
        default="bootstrap",
        help="Which optimizer to run (default: bootstrap)",
    )
    parser.add_argument(
        "--train-size",
        "-t",
        type=int,
        default=100,
        help="Number of training examples (default: 100)",
    )
    parser.add_argument(
        "--test-size",
        "-e",
        type=int,
        default=50,
        help="Number of test examples (default: 50)",
    )
    parser.add_argument(
        "--mipro-auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPRO optimization intensity (default: light)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable automatic plot generation",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (requires display)",
    )
    args = parser.parse_args()

    # Load environment and configure DSPy
    load_dotenv()
    lm = dspy.LM("openai/gpt-3.5-turbo")
    dspy.configure(lm=lm)

    print("=" * 60)
    print("Optimizer Runner")
    print("=" * 60)
    print(f"Optimizer: {args.optimizer}")
    print(f"Train size: {args.train_size}, Test size: {args.test_size}")

    # Load data
    print("\nLoading data...")
    trainset, testset = load_intent_classification_data(
        n_train=args.train_size, n_test=args.test_size
    )
    print(f"Loaded: {len(trainset)} train, {len(testset)} test")

    results = {}

    # Run selected optimizer(s)
    if args.optimizer in ["bootstrap", "both"]:
        optimized, result = run_bootstrap_fewshot(
            trainset=trainset,
            testset=testset,
            classifier_class=IntentClassifier,
            metric_fn=accuracy_metric,
            max_bootstrapped_demos=7,
            max_labeled_demos=7,
        )
        results["bootstrap"] = result
        output_path = Path("results/bootstrap_optimized.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        optimized.save(str(output_path))
        print(f"Saved to: {output_path}")

    if args.optimizer in ["mipro", "both"]:
        optimized, result = run_mipro_v2(
            trainset=trainset,
            testset=testset,
            classifier_class=IntentClassifier,
            metric_fn=accuracy_metric,
            auto=args.mipro_auto,
        )
        results["mipro"] = result
        output_path = Path("results/mipro_optimized.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        optimized.save(str(output_path))
        print(f"Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"{name}: {res}")

    # Generate visualizations
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        plot_optimization_results(
            results, output_dir="results", show_plots=args.show_plots
        )
