"""
Optimize Response Generator with MIPROv2

=== How MIPROv2 Works ===

MIPROv2 (Meta-optimizing Instructions through Prompting and Optimization) is an
advanced optimizer that discovers better INSTRUCTIONS and selects few-shot DEMOS.

1. Generates instruction candidates via meta-prompting (asks LLM to write instructions)

Meta-prompt to LLM:
"Given this task of generating customer support responses,
write 10 different instruction variations that could help
the model perform better. Here are some example inputs/outputs..."

LLM generates candidates like:
1. "Generate a helpful customer support response."
2. "You are an empathetic support agent. Provide clear, actionable responses..."
3. "Given the customer's intent, craft a professional response with next steps..."
...

2. For each candidate, combines with different few-shot demo selections
3. Evaluates each combination using your metric function (LLM-as-Judge)
4. Uses Bayesian optimization to efficiently search the space
5. Returns the best instruction + demo combination

    Meta-prompt ──> Instruction candidates ──> Combine with demos ──> Evaluate ──> Best combo
         │                   │                        │                  │              │
    [LLM writes]        [many options]          [+few-shot]         [metric()]    [Bayesian]

What gets optimized:
- Instructions: The task description ("system prompt" / signature docstring)
- Demos: Few-shot examples selected from training data

Auto modes (optimization intensity):
- light:  ~10-20 trials, 5-10 min, $1-5   (quick experiments)
- medium: ~30-50 trials, 20-40 min, $5-15 (better results)
- heavy:  ~100+ trials, hours, $20+       (maximum quality)

Docs: https://dspy.ai/api/optimizers/MIPROv2/

Usage:
    python examples/05_optimize_response_generator.py
    python examples/05_optimize_response_generator.py --auto medium
    python examples/05_optimize_response_generator.py --auto light --train-size 100 --test-size 50
"""

import argparse
import json
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import MIPROv2

from src.data.loader import load_query_data
from src.evaluation.llm_judge import strict_quality_metric
from src.evaluation.response_evaluator import evaluate
from src.modules.response_generator import (
    create_response_generator,
    get_optimized_instructions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize response generator with MIPROv2"
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Optimization intensity: light (~10 trials), medium (~30), heavy (~100+)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100,
        help="Training set size (default: 100)",
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


def main():
    args = parse_args()

    load_dotenv()
    lm = dspy.LM("openai/gpt-3.5-turbo")
    dspy.configure(lm=lm)

    print("RESPONSE GENERATOR OPTIMIZATION")
    print("=" * 60)
    print("Optimizer: MIPROv2")
    print(f"Settings: auto={args.auto}")
    print(f"Data: train={args.train_size}, test={args.test_size}")

    print("\n[1/5] Loading dataset...")
    trainset, testset = load_query_data(n_train=args.train_size, n_test=args.test_size)
    print(f"  Loaded {len(trainset)} train, {len(testset)} test examples")

    print("\n[2/5] Evaluating baseline...")
    baseline = create_response_generator()
    baseline_instructions = get_optimized_instructions(baseline)
    baseline_results = evaluate(baseline, testset, verbose=False)
    baseline_quality = baseline_results["average_quality"]
    print(f"  Baseline quality: {baseline_quality:.3f}")
    print(f"  Instructions: '{baseline_instructions}'")

    print(f"\n[3/5] Running MIPROv2 optimization (auto={args.auto})...")
    print("  This may take several minutes...")

    optimizer = MIPROv2(
        metric=strict_quality_metric,
        auto=args.auto,
        num_threads=4,
    )

    optimized = optimizer.compile(
        student=create_response_generator(),
        trainset=trainset,
    )
    print("  Optimization complete!")

    optimized_instructions = get_optimized_instructions(optimized)

    print("\n[4/5] Evaluating optimized generator...")
    optimized_results = evaluate(optimized, testset, verbose=False)
    optimized_quality = optimized_results["average_quality"]
    print(f"  Optimized quality: {optimized_quality:.3f}")

    improvement = optimized_quality - baseline_quality
    improvement_pct = (
        (improvement / baseline_quality) * 100 if baseline_quality > 0 else 0
    )

    print("\n[5/5] Saving results...")

    if args.name:
        exp_name = args.name
    else:
        exp_name = f"{args.auto}_tr{args.train_size}"

    results_dir = Path("results/response_generation")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": exp_name,
        "task": "response_generation",
        "optimizer": "MIPROv2",
        "optimizer_settings": {
            "auto": args.auto,
        },
        "dataset_sizes": {
            "train": len(trainset),
            "test": len(testset),
        },
        "baseline": {
            "average_quality": baseline_quality,
            "min_quality": baseline_results["min_quality"],
            "max_quality": baseline_results["max_quality"],
            "instructions": baseline_instructions,
        },
        "optimized": {
            "average_quality": optimized_quality,
            "min_quality": optimized_results["min_quality"],
            "max_quality": optimized_results["max_quality"],
            "instructions": optimized_instructions,
        },
        "improvement": {
            "absolute": improvement,
            "percentage": improvement_pct,
        },
    }

    results_path = results_dir / f"optimization_{exp_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    latest_path = results_dir / "optimization.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    model_path = results_dir / f"optimized_model_{exp_name}.json"
    optimized.save(str(model_path))

    print(f"  Results saved to {results_path}")
    print(f"  Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Baseline quality:  {baseline_quality:.3f}")
    print(f"Optimized quality: {optimized_quality:.3f}")
    print(f"Improvement:       {improvement:+.3f} ({improvement_pct:+.1f}%)")
    print()
    print("Instructions discovered:")
    print(f"  BEFORE: '{baseline_instructions}'")
    print(f"  AFTER:  '{optimized_instructions[:100]}...'")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
