"""
Advanced Optimization Strategies for Math Solver

Demonstrates MIPROv2 Medium optimization - a balanced approach for production use.
"""

from dotenv import load_dotenv
import dspy
from dspy.teleprompt import MIPROv2
import json
from pathlib import Path
import time

from src.data.loader import load_math_problems
from src.modules.math_solver import (
    create_math_solver,
    math_accuracy_metric,
)

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

print("ADVANCED OPTIMIZATION STRATEGIES")
print("=" * 50)


def evaluate_solver(solver, testset, description):
    """Evaluate a solver and return accuracy."""
    correct = 0
    for example in testset:
        try:
            pred = solver(question=example.question)
            if math_accuracy_metric(example, pred):
                correct += 1
        except Exception as e:
            print(f"  Warning: Error on example - {e}")
            continue

    accuracy = (correct / len(testset)) * 100
    print(f"  {description}: {accuracy:.1f}% ({correct}/{len(testset)})")
    return accuracy, correct


# Load dataset
print("\n[1] Loading datasets...")
medium_train, testset = load_math_problems(n_train=300, n_test=50)
print(f"  Train: {len(medium_train)} | Test: {len(testset)}")

# Baseline
print("\n[2] Establishing baseline...")
baseline = create_math_solver()
baseline_acc, baseline_correct = evaluate_solver(
    baseline, testset, "Baseline (zero-shot)"
)

# Store results
results = {
    "baseline": {"accuracy": baseline_acc, "correct": baseline_correct},
    "optimizations": [],
}

# MIPROv2 Medium optimization
print("\n[3] Running MIPROv2 Medium optimization...")
print("  Settings: 20 trials, 6 instruction candidates, 8 few-shot candidates")

start = time.time()
optimizer_medium = MIPROv2(
    metric=math_accuracy_metric,
    auto="medium",
    num_threads=4,
)
optimized_medium = optimizer_medium.compile(
    student=create_math_solver(),
    trainset=medium_train,
)
medium_time = time.time() - start
medium_acc, medium_correct = evaluate_solver(
    optimized_medium, testset, "MIPROv2 Medium"
)
results["optimizations"].append(
    {
        "strategy": "MIPROv2 Medium",
        "accuracy": medium_acc,
        "correct": medium_correct,
        "time_seconds": medium_time,
        "trainset_size": len(medium_train),
    }
)

# Summary
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Baseline:        {baseline_acc:.1f}%")
print(f"MIPROv2 Medium:  {medium_acc:.1f}% ({medium_acc - baseline_acc:+.1f}%) - Time: {medium_time:.1f}s")

# Save results
results_dir = Path("results/math_solver")
results_dir.mkdir(parents=True, exist_ok=True)
results_path = results_dir / "medium_optimization.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_path}")
