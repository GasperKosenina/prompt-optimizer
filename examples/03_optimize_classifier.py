"""
Optimize Intent Classifier with BootstrapFewShot
Fast and cheap

=== How BootstrapFewShot Works ===

1. Takes your training examples
2. Runs the model on each example at temperature=1.0 (for diverse outputs)
3. Evaluates each prediction using your metric function (e.g., accuracy_metric)
4. Keeps examples that pass the metric threshold as "bootstrapped demos"
5. Combines with "labeled demos" (ground-truth examples from training set)
6. The optimized prompt now includes these demos as few-shot examples

    Training Data ──> Run at temp=1.0 ──> Evaluate with metric ──> Keep passing ──> Few-shot prompt
                           │                      │                     │
                       [predict]            [metric(pred, gold)]    [demos]

Two types of demos:
- Bootstrapped demos: Model-generated examples that passed the metric (shows model's reasoning)
- Labeled demos: Raw examples from training set (ground truth input-output pairs)

Parameters:
- max_bootstrapped_demos: Max model-generated demos to include (default: 4)
- max_labeled_demos: Max ground-truth demos from trainset (default: 16)
- max_rounds: Iterations of bootstrapping for more diverse demos (default: 1)

Docs: https://dspy.ai/api/optimizers/BootstrapFewShot/

Usage:
    python examples/03_optimize_classifier.py
    python examples/03_optimize_classifier.py --demos 8 --rounds 2
    python examples/03_optimize_classifier.py --demos 4 --rounds 1 --train-size 500 --test-size 100
"""

import argparse
import dspy
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv
import json
from pathlib import Path

from src.data.loader import load_intent_classification_data
from src.modules.intent_classifier import (
    IntentClassifier,
    create_classifier,
    accuracy_metric,
    evaluate,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize intent classifier with BootstrapFewShot"
    )
    parser.add_argument(
        "--demos",
        type=int,
        default=4,
        help="Max bootstrapped & labeled demos (default: 4)",
    )
    parser.add_argument(
        "--rounds", type=int, default=1, help="Max optimization rounds (default: 1)"
    )
    parser.add_argument(
        "--train-size", type=int, default=200, help="Training set size (default: 200)"
    )
    parser.add_argument(
        "--test-size", type=int, default=50, help="Test set size (default: 50)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    return parser.parse_args()


def extract_demos(optimized_module):
    """Extract the few-shot demos from the optimized module."""
    demos = []
    for name, predictor in optimized_module.named_predictors():
        if hasattr(predictor, "demos") and predictor.demos:
            for demo in predictor.demos:
                demos.append(
                    {
                        "query": getattr(demo, "query", ""),
                        "intent": getattr(demo, "intent", ""),
                    }
                )
    return demos


def main():
    args = parse_args()

    load_dotenv()
    lm = dspy.LM("openai/gpt-3.5-turbo")
    dspy.configure(lm=lm)

    print("INTENT CLASSIFIER OPTIMIZATION")
    print("=" * 50)
    print("Optimizer: BootstrapFewShot")
    print(f"Settings: demos={args.demos}, rounds={args.rounds}")
    print(f"Data: train={args.train_size}, test={args.test_size}")

    print("\n[1/4] Loading Bitext dataset...")
    trainset, testset = load_intent_classification_data(
        n_train=args.train_size, n_test=args.test_size
    )
    print(f"  Loaded {len(trainset)} train, {len(testset)} test examples")

    print("\n[2/4] Evaluating baseline...")
    baseline = create_classifier()
    baseline_results = evaluate(baseline, testset, verbose=False)
    baseline_acc = baseline_results["accuracy"]
    print(f"  Baseline accuracy: {baseline_acc:.1f}%")

    print("\n[3/4] Running BootstrapFewShot optimization...")
    print(f"  max_bootstrapped_demos={args.demos}")
    print(f"  max_labeled_demos={args.demos}")
    print(f"  max_rounds={args.rounds}")

    optimizer = BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=args.demos,
        max_labeled_demos=args.demos,
        max_rounds=args.rounds,
    )

    optimized = optimizer.compile(
        student=dspy.Predict(IntentClassifier),
        trainset=trainset,
    )
    print("  Optimization complete!")

    print("\n[4/4] Evaluating optimized classifier...")
    optimized_results = evaluate(optimized, testset, verbose=False)
    optimized_acc = optimized_results["accuracy"]
    print(f"  Optimized accuracy: {optimized_acc:.1f}%")

    improvement = optimized_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0

    demos = extract_demos(optimized)
    print(f"  Selected {len(demos)} demos for few-shot prompt")

    if args.name:
        exp_name = args.name
    else:
        exp_name = f"d{args.demos}_r{args.rounds}_tr{args.train_size}"

    results_dir = Path("results/intent_classification")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": exp_name,
        "task": "intent_classification",
        "dataset": "bitext_27k",
        "baseline": {
            "accuracy": baseline_acc,
            "correct": baseline_results["correct"],
            "total": baseline_results["total"],
        },
        "optimized": {
            "accuracy": optimized_acc,
            "correct": optimized_results["correct"],
            "total": optimized_results["total"],
        },
        "improvement": {
            "absolute": improvement,
            "percentage": improvement_pct,
        },
        "dataset_sizes": {
            "train": len(trainset),
            "test": len(testset),
        },
        "optimizer": "BootstrapFewShot",
        "optimizer_settings": {
            "max_bootstrapped_demos": args.demos,
            "max_labeled_demos": args.demos,
            "max_rounds": args.rounds,
        },
        "selected_demos": demos,
    }

    results_path = results_dir / f"optimization_{exp_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    latest_path = results_dir / "optimization.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    optimized_path = results_dir / f"optimized_model_{exp_name}.json"
    optimized.save(str(optimized_path))

    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Experiment: {exp_name}")
    print(f"Baseline accuracy:  {baseline_acc:.1f}%")
    print(f"Optimized accuracy: {optimized_acc:.1f}%")
    print(f"Improvement:        {improvement:+.1f}% ({improvement_pct:+.1f}%)")

    if demos:
        print(f"\nSelected demos ({len(demos)}):")
        for i, demo in enumerate(demos[:5], 1):  # Show first 5
            print(f'  {i}. "{demo["query"][:50]}..." -> {demo["intent"]}')
        if len(demos) > 5:
            print(f"  ... and {len(demos) - 5} more")

    print(f"\nResults saved to {results_path}")
    print(f"Model saved to {optimized_path}")

    return results


if __name__ == "__main__":
    main()
