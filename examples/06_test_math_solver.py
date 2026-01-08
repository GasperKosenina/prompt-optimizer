"""
Test Math Solver - Baseline Calibration

Quick test of baseline math solving capability.
Goal: Verify everything works and check baseline accuracy (expected: 40-50%)
"""

from dotenv import load_dotenv
import dspy

from src.data.loader import load_math_problems
from src.modules.math_solver import (
    create_math_solver,
    extract_answer,
    answers_match,
)

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

print("=" * 60)
print("BASELINE MATH SOLVER TEST")
print("=" * 60)

# Load data
print("\nLoading GSM8K dataset...")
trainset, testset = load_math_problems(n_train=100, n_test=50)
print(f"Loaded: {len(trainset)} train, {len(testset)} test")

# Use all test examples for reliable baseline
test_subset = testset  # Full 50 examples
print(f"Testing on {len(test_subset)} examples for full baseline\n")

# Create baseline solver
print("Creating baseline math solver...")
solver = create_math_solver()
print("✓ Baseline created (with Chain-of-Thought)")

# Evaluate
print("\n" + "=" * 60)
print("EVALUATING BASELINE")
print("=" * 60)

correct = 0
total = 0
results = []

for i, example in enumerate(test_subset):
    # Get prediction
    pred = solver(question=example.question)

    # Extract expected answer
    expected = extract_answer(example.answer)
    predicted = pred.answer

    # Check correctness
    is_correct = answers_match(expected, predicted)
    correct += int(is_correct)
    total += 1

    results.append(
        {
            "question": example.question,
            "expected": expected,
            "predicted": predicted,
            "reasoning": pred.reasoning,
            "correct": is_correct,
        }
    )

    # Show first 3 in detail
    if i < 3:
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Example {i+1}:")
        print(f"Question: {example.question[:80]}...")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        if not is_correct:
            print(f"Reasoning: {pred.reasoning[:100]}...")

accuracy = (correct / total) * 100

print("\n" + "=" * 60)
print(f"BASELINE RESULTS ({len(test_subset)} examples)")
print("=" * 60)
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.1f}%")

# Assessment
if accuracy < 30:
    print("\n📊 Very low baseline - task may be too hard or solver needs tuning")
elif accuracy < 50:
    print("\n📊 Low baseline - good! Plenty of room for optimization ✅")
elif accuracy < 70:
    print("\n📊 Moderate baseline - optimization should still help")
else:
    print("\n📊 High baseline - task may be too easy for dramatic improvement")

print("=" * 60)

# Next steps
print("\n💡 Baseline test complete!")
print(f"   Baseline accuracy established: {accuracy:.1f}%")
print(f"   Next: Run optimization with examples/07_optimize_math_solver.py")
