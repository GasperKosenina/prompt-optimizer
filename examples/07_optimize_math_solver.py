"""
Usage:
    python examples/07_optimize_math_solver.py
    python examples/07_optimize_math_solver.py --auto medium
    python examples/07_optimize_math_solver.py --auto light --train-size 200 --test-size 50
"""

import argparse
import json
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import MIPROv2

from src.data.loader import load_math_problems
from src.modules.math_solver import (
    create_math_solver,
    math_accuracy_metric,
    extract_answer,
    answers_match,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize math solver with MIPROv2")
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Optimization intensity: light (~10 trials), medium (~30), heavy (~100+)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=200,
        help="Training set size (default: 200)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=50,
        help="Test set size (default: 50)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated from settings)",
    )
    return parser.parse_args()


def evaluate_solver(solver, testset):
    """Evaluate solver and return accuracy + predictions."""
    correct = 0
    predictions = []

    for example in testset:
        pred = solver(question=example.question)
        expected = extract_answer(example.answer)
        is_correct = answers_match(expected, pred.answer)

        correct += int(is_correct)
        predictions.append(
            {
                "question": example.question,
                "expected": expected,
                "predicted": pred.answer,
                "reasoning": pred.reasoning,
                "correct": is_correct,
            }
        )

    accuracy = (correct / len(testset)) * 100
    return accuracy, correct, predictions


def main():
    args = parse_args()

    load_dotenv()
    lm = dspy.LM("openai/gpt-3.5-turbo")
    dspy.configure(lm=lm)

    print("MATH SOLVER OPTIMIZATION")
    print("=" * 60)
    print("Optimizer: MIPROv2")
    print(f"Settings: auto={args.auto}")
    print(f"Data: train={args.train_size}, test={args.test_size}")

    print("\n[1/5] Loading GSM8K dataset...")
    trainset, testset = load_math_problems(
        n_train=args.train_size, n_test=args.test_size
    )
    print(f"  Loaded {len(trainset)} train, {len(testset)} test examples")

    print("\n[2/5] Evaluating baseline...")
    baseline = create_math_solver()
    baseline_acc, baseline_correct, baseline_preds = evaluate_solver(baseline, testset)
    print(
        f"  Baseline accuracy: {baseline_acc:.1f}% ({baseline_correct}/{len(testset)})"
    )

    print(f"\n[3/5] Running MIPROv2 optimization (auto={args.auto})...")
    print("  This may take several minutes...")

    optimizer = MIPROv2(
        metric=math_accuracy_metric,
        auto=args.auto,
        num_threads=4,
    )

    optimized = optimizer.compile(
        student=create_math_solver(),
        trainset=trainset,
    )
    print("  Optimization complete!")

    print("\n[4/5] Evaluating optimized solver...")
    optimized_acc, optimized_correct, optimized_preds = evaluate_solver(
        optimized, testset
    )
    print(
        f"  Optimized accuracy: {optimized_acc:.1f}% ({optimized_correct}/{len(testset)})"
    )

    improvement = optimized_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0

    print("\n[5/5] Saving results...")

    if args.name:
        exp_name = args.name
    else:
        exp_name = f"{args.auto}_tr{args.train_size}"

    results_dir = Path("results/math_solver")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": exp_name,
        "task": "math_word_problems",
        "dataset": "gsm8k",
        "optimizer": "MIPROv2",
        "optimizer_settings": {
            "auto": args.auto,
        },
        "dataset_sizes": {
            "train": len(trainset),
            "test": len(testset),
        },
        "baseline": {
            "accuracy": baseline_acc,
            "correct": baseline_correct,
            "total": len(testset),
        },
        "optimized": {
            "accuracy": optimized_acc,
            "correct": optimized_correct,
            "total": len(testset),
        },
        "improvement": {
            "absolute": improvement,
            "percentage": improvement_pct,
        },
    }

    results_path = results_dir / f"optimization_{exp_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    model_path = results_dir / f"optimized_model_{exp_name}.json"
    optimized.save(str(model_path))

    print(f"  Results saved to {results_path}")
    print(f"  Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Baseline accuracy:  {baseline_acc:.1f}%")
    print(f"Optimized accuracy: {optimized_acc:.1f}%")
    print(f"Improvement:        {improvement:+.1f}% ({improvement_pct:+.1f}%)")

    # Improvement analysis
    fixed = sum(
        1
        for i in range(len(testset))
        if optimized_preds[i]["correct"] and not baseline_preds[i]["correct"]
    )
    regressed = sum(
        1
        for i in range(len(testset))
        if not optimized_preds[i]["correct"] and baseline_preds[i]["correct"]
    )

    print(f"\nProblems fixed: {fixed}")
    print(f"Problems regressed: {regressed}")
    print(f"Net improvement: {fixed - regressed} problems")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
