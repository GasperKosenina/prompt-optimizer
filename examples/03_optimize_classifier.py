"""
Optimizing Intent Classifier with BootstrapFewShot

This script demonstrates DSPy optimization:
1. Define training examples with ground truth labels
2. Define a metric function (accuracy)
3. Use BootstrapFewShot to automatically find good few-shot examples
4. Compare baseline vs optimized performance
"""

import random
import dspy
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Configure DSPy
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


class IntentClassifier(dspy.Signature):
    """Classify customer support query into an intent category."""

    query: str = dspy.InputField(desc="Customer support query")
    intent: str = dspy.OutputField(
        desc="Intent category: REFUND, ORDER_STATUS, COMPLAINT, RETURN, BILLING, SHIPPING, PRODUCT_INFO"
    )


# Training data - examples with ground truth labels
TRAIN_DATA = [
    # REFUND examples
    ("I want my money back", "REFUND"),
    ("Please refund my purchase", "REFUND"),
    ("I need a refund for order #123", "REFUND"),
    ("Can I get my money back?", "REFUND"),
    # ORDER_STATUS examples
    ("Where is my order?", "ORDER_STATUS"),
    ("Track my package", "ORDER_STATUS"),
    ("When will my order arrive?", "ORDER_STATUS"),
    ("What's the status of my delivery?", "ORDER_STATUS"),
    # COMPLAINT examples
    ("This is unacceptable", "COMPLAINT"),
    ("I'm very disappointed with your service", "COMPLAINT"),
    ("Your product is terrible", "COMPLAINT"),
    ("I want to speak to a manager", "COMPLAINT"),
    # RETURN examples
    ("How do I return this?", "RETURN"),
    ("I want to send this back", "RETURN"),
    ("What's your return policy?", "RETURN"),
    ("Can I exchange this item?", "RETURN"),
    # BILLING examples
    ("I was charged twice", "BILLING"),
    ("There's an error on my bill", "BILLING"),
    ("Why was I charged extra?", "BILLING"),
    ("I don't recognize this charge", "BILLING"),
    # SHIPPING examples
    ("Do you ship internationally?", "SHIPPING"),
    ("What are the shipping costs?", "SHIPPING"),
    ("How long does delivery take?", "SHIPPING"),
    ("Can I get express shipping?", "SHIPPING"),
    # PRODUCT_INFO examples
    ("What colors is this available in?", "PRODUCT_INFO"),
    ("Is this compatible with my device?", "PRODUCT_INFO"),
    ("What are the dimensions?", "PRODUCT_INFO"),
    ("Does this come with a warranty?", "PRODUCT_INFO"),
]

# Test data - separate from training!
TEST_DATA = [
    ("Give me my money back now", "REFUND"),
    ("I demand a full refund", "REFUND"),
    ("Where is my package?", "ORDER_STATUS"),
    ("Has my order shipped yet?", "ORDER_STATUS"),
    ("This is the worst service ever", "COMPLAINT"),
    ("I'm filing a complaint", "COMPLAINT"),
    ("I need to return this product", "RETURN"),
    ("How can I send this back?", "RETURN"),
    ("You overcharged me", "BILLING"),
    ("Fix my invoice please", "BILLING"),
    ("Do you offer free shipping?", "SHIPPING"),
    ("Ship to Canada?", "SHIPPING"),
    ("What size should I order?", "PRODUCT_INFO"),
    ("Tell me about this product", "PRODUCT_INFO"),
]


def create_examples(data):
    """Convert tuples to DSPy Example objects."""
    return [
        dspy.Example(query=query, intent=intent).with_inputs("query")
        for query, intent in data
    ]


def accuracy_metric(example, prediction, trace=None):
    """Check if predicted intent matches expected intent (case-insensitive)."""
    return example.intent.lower() == prediction.intent.lower()


def evaluate_classifier(classifier, testset, name="Classifier"):
    """Evaluate a classifier on a test set and return accuracy."""
    correct = 0
    total = len(testset)

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}")
    print(f"{'=' * 60}")

    for example in testset:
        prediction = classifier(query=example.query)
        is_correct = accuracy_metric(example, prediction)
        correct += int(is_correct)

        status = "✓" if is_correct else "✗"
        print(f"{status} Query: {example.query[:40]:<40}")
        print(f"  Expected: {example.intent:<15} Got: {prediction.intent}")

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
    return accuracy


def main():
    # Convert data to DSPy Examples
    trainset = create_examples(TRAIN_DATA)
    testset = create_examples(TEST_DATA)

    print(f"Training examples: {len(trainset)}")
    print(f"Test examples: {len(testset)}")

    # Create baseline classifier
    baseline = dspy.Predict(IntentClassifier)

    # Evaluate baseline
    baseline_acc = evaluate_classifier(baseline, testset, "Baseline (zero-shot)")

    # Create optimizer
    print("\n" + "=" * 60)
    print("Optimizing with BootstrapFewShot...")
    print("=" * 60)

    # Shuffle training data to get diverse examples (not just first category)
    random.seed(42)
    shuffled_trainset = trainset.copy()
    random.shuffle(shuffled_trainset)

    optimizer = BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=7,  # One per intent category
        max_labeled_demos=7,  # One per intent category
        max_rounds=1,  # Keep low for budget
    )

    # Compile (optimize) the classifier
    optimized = optimizer.compile(
        student=dspy.Predict(IntentClassifier),
        trainset=shuffled_trainset,  # Use shuffled data
    )

    # Evaluate optimized classifier
    optimized_acc = evaluate_classifier(optimized, testset, "Optimized (few-shot)")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline accuracy:  {baseline_acc:.1f}%")
    print(f"Optimized accuracy: {optimized_acc:.1f}%")
    print(f"Improvement:        {optimized_acc - baseline_acc:+.1f}%")

    # Save optimized classifier
    optimized.save("results/optimized_intent_classifier.json")
    print("\nOptimized classifier saved to: results/optimized_intent_classifier.json")


if __name__ == "__main__":
    main()
