"""
Optimize Response Generator with MIPROv2

This script demonstrates prompt optimization for response generation:
1. Load dataset and create baseline generator
2. Evaluate baseline quality
3. Run MIPROv2 optimization (advanced meta-prompting)
4. Evaluate optimized generator
5. Compare results and inspect discovered instructions
"""

from dotenv import load_dotenv
import dspy
from dspy.teleprompt import MIPROv2
import json
from pathlib import Path

from src.data.loader import load_query_data
from src.modules.response_generator import (
    create_response_generator,
    get_optimized_instructions,
)
from src.evaluation.response_evaluator import evaluate
from src.evaluation.llm_judge import strict_quality_metric

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

print("=" * 70)
print("RESPONSE GENERATOR OPTIMIZATION")
print("=" * 70)
print("Optimizer: MIPROv2 (advanced meta-prompting)")
print("Goal: Discover improved instructions through prompt optimization")
print("=" * 70)

# Load dataset (small for fast iteration)
print("\n[1/6] Loading dataset...")
# Note: Need at least 27 examples for stratified sampling (27 intent categories)
trainset, testset = load_query_data(n_train=100, n_test=50)
print(f"  ✓ Loaded {len(trainset)} train, {len(testset)} test examples")

# Create baseline generator
print("\n[2/6] Creating baseline generator...")
baseline = create_response_generator()
baseline_instructions = get_optimized_instructions(baseline)
print(f"  ✓ Baseline created")
print(f"  Instructions: '{baseline_instructions}'")

# Evaluate baseline
print("\n[3/6] Evaluating baseline...")
baseline_results = evaluate(baseline, testset, verbose=False)
baseline_quality = baseline_results["average_quality"]
print(f"  ✓ Baseline quality: {baseline_quality:.3f}")
print(
    f"  Range: {baseline_results['min_quality']:.3f} - {baseline_results['max_quality']:.3f}"
)

# Optimize with MIPROv2
print("\n[4/6] Running MIPROv2 optimization...")
print("  (This may take 5-10 minutes...)")
print("  MIPROv2 uses meta-prompting to discover better instructions")

optimizer = MIPROv2(
    metric=strict_quality_metric,
    auto="light",  # Use "light" mode for faster optimization
    num_threads=4,  # Parallel evaluation for speed
)

# Compile (optimize) the generator
optimized = optimizer.compile(
    student=create_response_generator(),
    trainset=trainset,
)

print("  ✓ Optimization complete!")

# Inspect optimized instructions
optimized_instructions = get_optimized_instructions(optimized)
print(f"\n  Optimized instructions: '{optimized_instructions}'")

# Evaluate optimized generator
print("\n[5/6] Evaluating optimized generator...")
optimized_results = evaluate(optimized, testset, verbose=False)
optimized_quality = optimized_results["average_quality"]
print(f"  ✓ Optimized quality: {optimized_quality:.3f}")
print(
    f"  Range: {optimized_results['min_quality']:.3f} - {optimized_results['max_quality']:.3f}"
)

# Compare results
print("\n[6/6] Saving results...")

improvement = optimized_quality - baseline_quality
improvement_pct = (improvement / baseline_quality) * 100

results = {
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
    "dataset": {
        "train_size": len(trainset),
        "test_size": len(testset),
    },
}

# Save results to JSON
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

results_path = results_dir / "response_generator_optimization.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"  ✓ Results saved to {results_path}")

# Save optimized module
optimized_path = results_dir / "response_generator_optimized.json"
optimized.save(str(optimized_path))
print(f"  ✓ Optimized module saved to {optimized_path}")

# Print summary
print("\n" + "=" * 70)
print("OPTIMIZATION RESULTS SUMMARY")
print("=" * 70)
print(f"Baseline quality:     {baseline_quality:.3f}")
print(f"Optimized quality:    {optimized_quality:.3f}")
print(f"Improvement:          {improvement:+.3f} ({improvement_pct:+.1f}%)")
print()
print("Instructions discovered:")
print(f"  Before: '{baseline_instructions}'")
print(f"  After:  '{optimized_instructions}'")
print("=" * 70)

# Show side-by-side comparison for first 3 examples
print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON (First 3 Examples)")
print("=" * 70)

for i in range(min(3, len(testset))):
    example = testset[i]

    # Generate with both
    baseline_pred = baseline(query=example.query, intent=example.intent)
    optimized_pred = optimized(query=example.query, intent=example.intent)

    # Get scores
    baseline_score = baseline_results["scores"][i]["quality_score"]
    optimized_score = optimized_results["scores"][i]["quality_score"]

    print(f"\n--- Example {i+1} ---")
    print(f"Query: {example.query[:80]}...")
    print(f"Intent: {example.intent}")
    print()
    print(f"BASELINE (Score: {baseline_score:.2f}):")
    print(f"  {baseline_pred.response[:120]}...")
    print()
    print(f"OPTIMIZED (Score: {optimized_score:.2f}):")
    print(f"  {optimized_pred.response[:120]}...")
    print()
    improvement_example = optimized_score - baseline_score
    if improvement_example > 0:
        print(f"  ✅ Improvement: +{improvement_example:.2f}")
    elif improvement_example < 0:
        print(f"  ❌ Regression: {improvement_example:.2f}")
    else:
        print(f"  ➖ No change")

print("\n" + "=" * 70)
print("Optimization complete! Check results/ directory for saved outputs.")
print("=" * 70)
