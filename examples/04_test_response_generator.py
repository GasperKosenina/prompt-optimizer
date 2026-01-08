"""
Test Response Generator - Baseline Calibration

This script tests the baseline response generator with strict quality evaluation.
Goal: Calibrate judges so baseline scores around 0.65-0.75.
"""

from dotenv import load_dotenv
import dspy

from src.data.loader import load_query_data
from src.modules.response_generator import (
    create_response_generator,
    get_optimized_instructions,
)
from src.evaluation.response_evaluator import evaluate

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

print("=" * 60)
print("BASELINE RESPONSE GENERATOR TEST")
print("=" * 60)

# Load data WITHOUT gold responses (principle-based evaluation)
print("\nLoading data...")
trainset, testset = load_query_data(n_train=100, n_test=50)
print(f"Loaded: {len(trainset)} train, {len(testset)} test")
print(f"Testing on first 10 examples for calibration check\n")

# Create baseline generator
print("Creating baseline generator...")
baseline = create_response_generator()

# Show what instructions the baseline uses
baseline_instructions = get_optimized_instructions(baseline)
print("\nBaseline instructions (from signature):")
print(f"  '{baseline_instructions}'")
print()

# Evaluate with strict judge
print("=" * 60)
print("EVALUATING WITH STRICT JUDGE")
print("=" * 60)
print("Target baseline range: 0.65-0.75")
print()

results = evaluate(baseline, testset[:10], verbose=True)

# Show sample responses for inspection
print("\n" + "=" * 60)
print("SAMPLE RESPONSES (First 3)")
print("=" * 60)
for i, score in enumerate(results["scores"][:3], 1):
    print(f"\n--- Response {i} (Quality: {score['quality_score']:.2f}) ---")
    print(f"Query: {score['query'][:80]}...")
    print(f"Intent: {score['intent']}")
    print(f"\nGenerated Response:")
    print(f"  {score['response'][:150]}...")
    print(f"\nJudge's Reasoning:")
    print(f"  {score['reasoning'][:120]}...")

# Final calibration check
print("\n" + "=" * 60)
print("CALIBRATION CHECK")
print("=" * 60)
avg = results["average_quality"]
print(f"Average baseline quality: {avg:.3f}")

if 0.65 <= avg <= 0.75:
    print("✅ CALIBRATED! Baseline is in target range (0.65-0.75)")
    print("   Ready to proceed with optimization.")
elif avg < 0.65:
    print("⚠️  Baseline too low. Judge may be too strict.")
    print("   Consider adjusting scoring guidelines.")
elif avg > 0.75:
    print("⚠️  Baseline too high. Judge is too lenient.")
    print("   Need stricter evaluation criteria.")

print("=" * 60)
