"""
Intent Classification with DSPy

Demonstrates classifying customer support queries into 27 intent categories
using the Bitext dataset. This establishes a baseline before optimization.
"""

import dspy
from dotenv import load_dotenv

from src.data.loader import load_intent_classification_data
from src.modules.intent_classifier import (
    create_classifier,
    evaluate,
    INTENT_LABELS,
)

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

print("INTENT CLASSIFICATION BASELINE")
print("=" * 50)
print(f"Task: Classify queries into {len(INTENT_LABELS)} intent categories")

# Load dataset
print("\n[1/2] Loading Bitext dataset...")
trainset, testset = load_intent_classification_data(n_train=100, n_test=50)
print(f"  Loaded {len(trainset)} train, {len(testset)} test examples")

# Create baseline classifier
print("\n[2/2] Evaluating baseline classifier...")
classifier = create_classifier()
results = evaluate(classifier, testset, verbose=False)

# Summary
print("\n" + "=" * 50)
print("BASELINE RESULTS")
print("=" * 50)
print(f"Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")

# Show a few example predictions
print("\nSample predictions:")
for pred in results["predictions"][:5]:
    status = "✓" if pred["correct"] else "✗"
    print(f"  {status} '{pred['query'][:40]}...' → {pred['predicted']}")

print("\nNext: Run 03_optimize_classifier.py to improve this baseline")
