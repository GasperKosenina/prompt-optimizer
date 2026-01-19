"""
Optimize Intent Classifier with BootstrapFewShot

Demonstrates DSPy optimization on intent classification:
1. Establish baseline accuracy
2. Use BootstrapFewShot to find good few-shot examples
3. Compare baseline vs optimized performance
4. Save results for analysis
"""

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

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

print("INTENT CLASSIFIER OPTIMIZATION")
print("=" * 50)
print("Optimizer: BootstrapFewShot")

# Load dataset
print("\n[1/4] Loading Bitext dataset...")
trainset, testset = load_intent_classification_data(n_train=200, n_test=50)
print(f"  Loaded {len(trainset)} train, {len(testset)} test examples")

# Evaluate baseline
print("\n[2/4] Evaluating baseline...")
baseline = create_classifier()
baseline_results = evaluate(baseline, testset, verbose=False)
baseline_acc = baseline_results["accuracy"]
print(f"  Baseline accuracy: {baseline_acc:.1f}%")

# Optimize
print("\n[3/4] Running BootstrapFewShot optimization...")
optimizer = BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    max_rounds=1,
)

optimized = optimizer.compile(
    student=dspy.Predict(IntentClassifier),
    trainset=trainset,
)
print("  Optimization complete!")

# Evaluate optimized
print("\n[4/4] Evaluating optimized classifier...")
optimized_results = evaluate(optimized, testset, verbose=False)
optimized_acc = optimized_results["accuracy"]
print(f"  Optimized accuracy: {optimized_acc:.1f}%")

# Calculate improvement
improvement = optimized_acc - baseline_acc
improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0

# Save results
results_dir = Path("results/intent_classification")
results_dir.mkdir(parents=True, exist_ok=True)

results = {
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
}

results_path = results_dir / "optimization.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

optimized_path = results_dir / "optimized_model.json"
optimized.save(str(optimized_path))

# Summary
print("\n" + "=" * 50)
print("OPTIMIZATION RESULTS")
print("=" * 50)
print(f"Baseline accuracy:  {baseline_acc:.1f}%")
print(f"Optimized accuracy: {optimized_acc:.1f}%")
print(f"Improvement:        {improvement:+.1f}% ({improvement_pct:+.1f}%)")
print(f"\nResults saved to {results_path}")
print(f"Model saved to {optimized_path}")
