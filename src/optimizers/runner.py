"""
DSPy Optimizer Runner

Runs and compares DSPy optimizers (BootstrapFewShot and MIPROv2).
Supports both classification and response generation tasks.
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

from src.optimizers.evaluator import (  # type: ignore[import-not-found]
    compute_classification_metrics,
    print_metrics_summary,
    print_detailed_comparison,
)
from src.visualization.plots import plot_optimization_results  # type: ignore[import-not-found]


# ============================================================================
# Data Structures
# ============================================================================


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
    settings: dict

    # Classification metrics (baseline)
    baseline_precision_macro: float | None = None
    baseline_recall_macro: float | None = None
    baseline_f1_macro: float | None = None
    baseline_precision_weighted: float | None = None
    baseline_recall_weighted: float | None = None
    baseline_f1_weighted: float | None = None

    # Classification metrics (optimized)
    optimized_precision_macro: float | None = None
    optimized_recall_macro: float | None = None
    optimized_f1_macro: float | None = None
    optimized_precision_weighted: float | None = None
    optimized_recall_weighted: float | None = None
    optimized_f1_weighted: float | None = None

    # Cost tracking (optional)
    total_api_calls: int | None = None
    estimated_cost_usd: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def __str__(self) -> str:
        cost_str = f" [${self.estimated_cost_usd:.2f}]" if self.estimated_cost_usd else ""
        return (
            f"{self.optimizer_name}: "
            f"Acc: {self.baseline_accuracy:.1f}% → {self.optimized_accuracy:.1f}% "
            f"({self.improvement:+.1f}%), "
            f"F1: {self.baseline_f1_macro:.3f} → {self.optimized_f1_macro:.3f}"
            f"{cost_str}"
        )


# ============================================================================
# Helper Functions
# ============================================================================


def _print_progress(message: str) -> None:
    """Print a progress message and flush output."""
    print(message)
    sys.stdout.flush()


def _check_mipro_dataset_size(trainset_size: int, auto: str) -> bool:
    """
    Warn user if dataset size is large for MIPROv2 optimization.

    Returns True if user confirms to continue, False otherwise.
    """
    warnings = []

    if auto == "light" and trainset_size > 100:
        warnings.extend([
            f"⚠️  Large dataset ({trainset_size} examples) for MIPROv2 light mode",
            "   Recommended: 50-100 examples for light mode",
            "   This may take 10-30 minutes and cost $1-5",
        ])
    elif auto == "medium" and trainset_size > 300:
        warnings.extend([
            f"⚠️  Large dataset ({trainset_size} examples) for MIPROv2 medium mode",
            "   Recommended: 200-300 examples for medium mode",
            "   This may take 30-90 minutes and cost $5-20",
        ])
    elif auto == "heavy" and trainset_size > 100:
        warnings.extend([
            f"⚠️  Large dataset ({trainset_size} examples) for MIPROv2 heavy mode",
            "   Recommended: <100 examples for heavy mode",
            "   This may take HOURS and cost $20-100+",
        ])

    if warnings:
        print("\n" + "!" * 60)
        for warning in warnings:
            print(warning)
        print("!" * 60)

        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        return response in ["y", "yes"]

    return True


def _create_optimization_result(
    optimizer_name: str,
    baseline_metrics: dict,
    optimized_metrics: dict,
    train_size: int,
    test_size: int,
    duration: float,
    settings: dict,
) -> OptimizationResult:
    """Create an OptimizationResult from metrics dictionaries."""
    baseline_accuracy = baseline_metrics["accuracy"]
    optimized_accuracy = optimized_metrics["accuracy"]

    return OptimizationResult(
        optimizer_name=optimizer_name,
        baseline_accuracy=baseline_accuracy,
        optimized_accuracy=optimized_accuracy,
        improvement=optimized_accuracy - baseline_accuracy,
        train_size=train_size,
        test_size=test_size,
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
        settings=settings,
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


# ============================================================================
# Optimization Functions
# ============================================================================


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

    BootstrapFewShot selects good few-shot examples by running the model
    on training data and keeping examples where it succeeds.

    Args:
        trainset: Training examples
        testset: Test examples for evaluation
        classifier_class: DSPy Signature class (e.g., IntentClassifier)
        metric_fn: Function(example, prediction) -> bool
        max_bootstrapped_demos: Max auto-generated demos
        max_labeled_demos: Max labeled examples from trainset
        max_rounds: Number of optimization rounds
        verbose: Print progress

    Returns:
        Tuple of (optimized_classifier, OptimizationResult)
    """
    start_time = time.time()

    if verbose:
        print("=" * 60)
        print("Running BootstrapFewShot Optimization")
        print("=" * 60)
        print(f"Train size: {len(trainset)}, Test size: {len(testset)}")
        print(
            f"Settings: max_bootstrapped_demos={max_bootstrapped_demos}, "
            f"max_labeled_demos={max_labeled_demos}, max_rounds={max_rounds}"
        )

    # Step 1: Evaluate baseline
    if verbose:
        _print_progress("\n[1/4] Evaluating baseline classifier...")

    baseline = dspy.Predict(classifier_class)
    baseline_metrics = compute_classification_metrics(testset, baseline)

    if verbose:
        print_metrics_summary(baseline_metrics, "Baseline")

    # Step 2: Run optimization
    if verbose:
        _print_progress("\n[2/4] Running BootstrapFewShot optimization...")
        print(f"       • Bootstrapping from {len(trainset)} training examples")

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

    # Step 3: Evaluate optimized
    if verbose:
        _print_progress("\n[3/4] Evaluating optimized classifier...")

    optimized_metrics = compute_classification_metrics(testset, optimized)

    if verbose:
        print_metrics_summary(optimized_metrics, "Optimized")
        print(f"\n       Improvement: {optimized_metrics['accuracy'] - baseline_metrics['accuracy']:+.1f}%")
        print(f"       Duration: {time.time() - start_time:.1f}s")
        print_detailed_comparison(baseline_metrics, optimized_metrics)

    # Step 4: Create result
    result = _create_optimization_result(
        optimizer_name="BootstrapFewShot",
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        train_size=len(trainset),
        test_size=len(testset),
        duration=time.time() - start_time,
        settings={
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "max_rounds": max_rounds,
        },
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

    MIPROv2 generates and evaluates multiple prompt variations using
    Bayesian optimization to find the best instruction + few-shot combo.

    Args:
        trainset: Training examples
        testset: Test examples for evaluation
        classifier_class: DSPy Signature class
        metric_fn: Function(example, prediction) -> bool
        auto: Optimization intensity ("light", "medium", "heavy")
        num_threads: Parallel evaluation threads
        verbose: Print progress

    Returns:
        Tuple of (optimized_classifier, OptimizationResult)
    """
    start_time = time.time()

    if verbose:
        print("=" * 60)
        print("Running MIPROv2 Optimization")
        print("=" * 60)
        print(f"Train size: {len(trainset)}, Test size: {len(testset)}")
        print(f"Settings: auto={auto}, num_threads={num_threads}")
        print("\nNote: MIPROv2 may take several minutes and use more API calls.")

    # Check dataset size and get user confirmation
    if not _check_mipro_dataset_size(len(trainset), auto):
        print("\n❌ Optimization cancelled by user.")
        baseline = dspy.Predict(classifier_class)
        baseline_metrics = compute_classification_metrics(testset, baseline)

        result = _create_optimization_result(
            optimizer_name="MIPROv2",
            baseline_metrics=baseline_metrics,
            optimized_metrics=baseline_metrics,  # Same as baseline since cancelled
            train_size=len(trainset),
            test_size=len(testset),
            duration=0.0,
            settings={"auto": auto, "num_threads": num_threads, "cancelled": True},
        )
        return baseline, result

    # Step 1: Evaluate baseline
    if verbose:
        _print_progress("\n[1/3] Evaluating baseline...")

    baseline = dspy.Predict(classifier_class)
    baseline_metrics = compute_classification_metrics(testset, baseline)

    if verbose:
        print_metrics_summary(baseline_metrics, "Baseline")

    # Step 2: Run optimization
    if verbose:
        _print_progress("\n[2/3] Optimizing (this may take a while)...")

    optimizer = MIPROv2(
        metric=metric_fn,
        auto=auto,
        num_threads=num_threads,
    )

    optimized = optimizer.compile(
        student=dspy.Predict(classifier_class),
        trainset=trainset,
    )

    # Step 3: Evaluate optimized
    if verbose:
        _print_progress("\n[3/3] Evaluating optimized classifier...")

    optimized_metrics = compute_classification_metrics(testset, optimized)

    if verbose:
        print_metrics_summary(optimized_metrics, "Optimized")
        print(f"\n         Improvement: {optimized_metrics['accuracy'] - baseline_metrics['accuracy']:+.1f}%")
        print(f"         Duration: {time.time() - start_time:.1f}s")
        print_detailed_comparison(baseline_metrics, optimized_metrics)

    # Create result
    result = _create_optimization_result(
        optimizer_name="MIPROv2",
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        train_size=len(trainset),
        test_size=len(testset),
        duration=time.time() - start_time,
        settings={"auto": auto, "num_threads": num_threads},
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
        classifier_class: DSPy Signature class
        metric_fn: Metric function
        optimizers: List of optimizers to run (default: ["bootstrap", "mipro_light"])
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

        result: OptimizationResult
        if opt_name == "bootstrap":
            _, result = run_bootstrap_fewshot(
                trainset, testset, classifier_class, metric_fn, verbose=verbose
            )
        elif opt_name == "mipro_light":
            _, result = run_mipro_v2(
                trainset, testset, classifier_class, metric_fn, auto="light", verbose=verbose
            )
        elif opt_name == "mipro_medium":
            _, result = run_mipro_v2(
                trainset, testset, classifier_class, metric_fn, auto="medium", verbose=verbose
            )
        else:
            # Should never happen due to type hints, but handle gracefully
            print(f"Warning: Unknown optimizer '{opt_name}', skipping")
            continue

        results[opt_name] = result

    # Print comparison summary
    if verbose:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY - ACCURACY")
        print("=" * 80)
        print(
            f"{'Optimizer':<20} {'Baseline':>10} {'Optimized':>10} "
            f"{'Improvement':>12} {'Time':>10}"
        )
        print("-" * 80)

        for name, res in results.items():
            print(
                f"{name:<20} {res.baseline_accuracy:>9.1f}% "
                f"{res.optimized_accuracy:>9.1f}% "
                f"{res.improvement:>+11.1f}% {res.duration_seconds:>9.1f}s"
            )

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY - F1 SCORES (MACRO)")
        print("=" * 80)
        print(f"{'Optimizer':<20} {'Baseline F1':>12} {'Optimized F1':>13} {'Improvement':>12}")
        print("-" * 80)

        for name, res in results.items():
            baseline_f1 = res.baseline_f1_macro or 0
            optimized_f1 = res.optimized_f1_macro or 0
            improvement = optimized_f1 - baseline_f1
            print(
                f"{name:<20} {baseline_f1:>12.3f} {optimized_f1:>13.3f} "
                f"{improvement:>+12.3f}"
            )

        print("=" * 80)

    return results


# ============================================================================
# Results Management
# ============================================================================


def save_results(
    results: dict[str, OptimizationResult],
    filepath: str | Path,
) -> None:
    """Save optimization results to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {name: res.to_dict() for name, res in results.items()}

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {filepath}")


def load_results(filepath: str | Path) -> dict[str, OptimizationResult]:
    """Load optimization results from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    return {name: OptimizationResult(**res_dict) for name, res_dict in data.items()}


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_intent_classification_data  # type: ignore[import-not-found]
    from src.modules.intent_classifier import (  # type: ignore[import-not-found]
        IntentClassifier,
        accuracy_metric,
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run DSPy optimizers")
    parser.add_argument(
        "--optimizer", "-o",
        choices=["bootstrap", "mipro", "both"],
        default="bootstrap",
        help="Which optimizer to run",
    )
    parser.add_argument(
        "--train-size", "-t",
        type=int,
        default=100,
        help="Number of training examples",
    )
    parser.add_argument(
        "--test-size", "-e",
        type=int,
        default=50,
        help="Number of test examples",
    )
    parser.add_argument(
        "--mipro-auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPRO optimization intensity",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable automatic plot generation",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively",
    )
    args = parser.parse_args()

    # Setup
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

    # Run optimizers
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

        # Save optimized classifier
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

        # Save optimized classifier
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
        plot_optimization_results(results, output_dir="results", show_plots=args.show_plots)
