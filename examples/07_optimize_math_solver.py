"""
Optimize Math Solver with MIPROv2

Demonstrates prompt optimization for math reasoning:
1. Establish baseline accuracy
2. Run MIPROv2 to discover better reasoning patterns
3. Compare baseline vs optimized performance
4. Show example improvements
"""

from dotenv import load_dotenv
import dspy
from dspy.teleprompt import MIPROv2
import json
from pathlib import Path

from src.data.loader import load_math_problems
from src.modules.math_solver import (
    create_math_solver,
    math_accuracy_metric,
    extract_answer,
    answers_match,
)

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

print("MATH SOLVER OPTIMIZATION")
print("=" * 50)
print("Task: Grade-school math word problems (GSM8K)")

# Load dataset
print("\n[1/6] Loading GSM8K dataset...")
trainset, testset = load_math_problems(n_train=200, n_test=50)
print(f"  Loaded {len(trainset)} train, {len(testset)} test examples")

# Create baseline
print("\n[2/6] Creating baseline solver...")
baseline = create_math_solver()

# Evaluate baseline
print("\n[3/6] Evaluating baseline...")
baseline_correct = 0
baseline_predictions = []

for example in testset:
    pred = baseline(question=example.question)
    expected = extract_answer(example.answer)
    is_correct = answers_match(expected, pred.answer)

    baseline_correct += int(is_correct)
    baseline_predictions.append(
        {
            "question": example.question,
            "expected": expected,
            "predicted": pred.answer,
            "reasoning": pred.reasoning,
            "correct": is_correct,
        }
    )

baseline_accuracy = (baseline_correct / len(testset)) * 100
print(f"  Baseline accuracy: {baseline_accuracy:.1f}% ({baseline_correct}/{len(testset)})")

# Optimize
print("\n[4/6] Running MIPROv2 optimization...")
print("  (This may take 10-15 minutes...)")

optimizer = MIPROv2(
    metric=math_accuracy_metric,
    auto="light",
    num_threads=4,
)

optimized = optimizer.compile(
    student=create_math_solver(),
    trainset=trainset,
)

print("  Optimization complete!")

# Evaluate optimized
print("\n[5/6] Evaluating optimized solver...")
optimized_correct = 0
optimized_predictions = []

for example in testset:
    pred = optimized(question=example.question)
    expected = extract_answer(example.answer)
    is_correct = answers_match(expected, pred.answer)

    optimized_correct += int(is_correct)
    optimized_predictions.append(
        {
            "question": example.question,
            "expected": expected,
            "predicted": pred.answer,
            "reasoning": pred.reasoning,
            "correct": is_correct,
        }
    )

optimized_accuracy = (optimized_correct / len(testset)) * 100
print(f"  Optimized accuracy: {optimized_accuracy:.1f}% ({optimized_correct}/{len(testset)})")

# Save results
print("\n[6/6] Saving results...")

improvement = optimized_accuracy - baseline_accuracy
improvement_pct = (
    (improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
)

results = {
    "task": "math_word_problems",
    "dataset": "gsm8k",
    "baseline": {
        "accuracy": baseline_accuracy,
        "correct": baseline_correct,
        "total": len(testset),
    },
    "optimized": {
        "accuracy": optimized_accuracy,
        "correct": optimized_correct,
        "total": len(testset),
    },
    "improvement": {
        "absolute": improvement,
        "percentage": improvement_pct,
    },
    "dataset_sizes": {
        "train": len(trainset),
        "test": len(testset),
    },
}

results_dir = Path("results/math_solver")
results_dir.mkdir(parents=True, exist_ok=True)

results_path = results_dir / "light_optimization.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

optimized_path = results_dir / "optimized_model_light.json"
optimized.save(str(optimized_path))

print(f"  Results saved to {results_path}")
print(f"  Optimized model saved to {optimized_path}")

# Summary
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Baseline accuracy:    {baseline_accuracy:.1f}%")
print(f"Optimized accuracy:   {optimized_accuracy:.1f}%")
print(f"Improvement:          {improvement:+.1f}% ({improvement_pct:+.1f}%)")

# Show comparison examples
print("\n" + "=" * 50)
print("SIDE-BY-SIDE COMPARISON (First 5 Examples)")
print("=" * 50)

for i in range(min(5, len(testset))):
    b_pred = baseline_predictions[i]
    o_pred = optimized_predictions[i]

    print(f"\n--- Example {i+1} ---")
    print(f"Question: {b_pred['question'][:100]}...")
    print(f"Expected: {b_pred['expected']}")
    print(f"Baseline: {b_pred['predicted']} {'✅' if b_pred['correct'] else '❌'}")
    print(f"Optimized: {o_pred['predicted']} {'✅' if o_pred['correct'] else '❌'}")

    if o_pred["correct"] and not b_pred["correct"]:
        print("  Optimization fixed this!")
    elif not o_pred["correct"] and b_pred["correct"]:
        print("  Regression on this example")

# Improvement analysis
print("\n" + "=" * 50)
print("IMPROVEMENT ANALYSIS")
print("=" * 50)

fixed = sum(
    1
    for i in range(len(testset))
    if optimized_predictions[i]["correct"] and not baseline_predictions[i]["correct"]
)
regressed = sum(
    1
    for i in range(len(testset))
    if not optimized_predictions[i]["correct"] and baseline_predictions[i]["correct"]
)

print(f"Problems fixed by optimization: {fixed}")
print(f"Problems that regressed: {regressed}")
print(f"Net improvement: {fixed - regressed} problems")
print("\nOptimization complete!")
