"""
Intent Classification Module

This module defines the DSPy signature and utilities for classifying
customer support queries into one of 27 intent categories.

Task: Given a customer query, predict the intent.

Example:
    Input:  "I want to cancel my order #12345"
    Output: "cancel_order"
"""

import dspy

# All 27 intent labels from the Bitext dataset
INTENT_LABELS = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
]

# Format labels for the prompt description
INTENT_LABELS_STR = ", ".join(INTENT_LABELS)


class IntentClassifier(dspy.Signature):
    """
    Classify a customer support query into one of 27 intent categories.

    The intent represents what action or information the customer is seeking.
    Choose the most specific intent that matches the customer's request.
    """

    query: str = dspy.InputField(
        desc="A customer support query or message"
    )
    intent: str = dspy.OutputField(
        desc=f"The intent category. Must be one of: {INTENT_LABELS_STR}"
    )


def create_classifier() -> dspy.Predict:
    """
    Create a basic intent classifier using dspy.Predict.

    Returns:
        A DSPy Predict module configured with the IntentClassifier signature.

    Example:
        classifier = create_classifier()
        result = classifier(query="I want to cancel my order")
        print(result.intent)  # "cancel_order"
    """
    return dspy.Predict(IntentClassifier)


def accuracy_metric(example: dspy.Example, prediction: dspy.Prediction, _trace=None) -> bool:
    """
    Compute whether the predicted intent matches the expected intent.

    This is the metric function used by DSPy optimizers to evaluate predictions.

    Args:
        example: A DSPy Example with the ground truth 'intent' field
        prediction: A DSPy Prediction with the predicted 'intent' field
        trace: Optional trace information (unused, but required by DSPy)

    Returns:
        True if prediction matches expected intent (case-insensitive), False otherwise

    Example:
        example = dspy.Example(query="cancel order", intent="cancel_order")
        prediction = classifier(query="cancel order")
        is_correct = accuracy_metric(example, prediction)
    """
    expected = example.intent.lower().strip()
    predicted = prediction.intent.lower().strip()
    return expected == predicted


def evaluate(
    classifier: dspy.Predict,
    testset: list[dspy.Example],
    verbose: bool = True,
) -> dict:
    """
    Evaluate a classifier on a test set and return metrics.

    Args:
        classifier: A DSPy Predict module (baseline or optimized)
        testset: List of DSPy Examples with query and intent fields
        verbose: If True, print progress and results

    Returns:
        Dictionary with evaluation results:
        - accuracy: Percentage of correct predictions (0-100)
        - correct: Number of correct predictions
        - total: Total number of examples
        - predictions: List of (query, expected, predicted, is_correct) tuples

    Example:
        from src.data.loader import load_intent_classification_data
        trainset, testset = load_intent_classification_data(n_test=50)
        classifier = create_classifier()
        results = evaluate(classifier, testset)
        print(f"Accuracy: {results['accuracy']:.1f}%")
    """
    correct = 0
    total = len(testset)
    predictions = []

    if verbose:
        print(f"\nEvaluating on {total} examples...")
        print("-" * 60)

    for i, example in enumerate(testset):
        # Get prediction
        prediction = classifier(query=example.query)

        # Check if correct
        is_correct = accuracy_metric(example, prediction)
        correct += int(is_correct)

        # Store result
        predictions.append({
            "query": example.query,
            "expected": example.intent,
            "predicted": prediction.intent,
            "correct": is_correct,
        })

        # Print progress
        if verbose:
            status = "✓" if is_correct else "✗"
            # Truncate long queries for display
            query_display = example.query[:50] + "..." if len(example.query) > 50 else example.query
            print(f"{status} [{i+1}/{total}] {query_display}")
            if not is_correct:
                print(f"    Expected: {example.intent}, Got: {prediction.intent}")

    accuracy = (correct / total) * 100

    if verbose:
        print("-" * 60)
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def get_confusion_matrix(predictions: list[dict]) -> dict[str, dict[str, int]]:
    """
    Build a confusion matrix from prediction results.

    Args:
        predictions: List of prediction dicts from evaluate()

    Returns:
        Nested dict where confusion[expected][predicted] = count

    Example:
        results = evaluate(classifier, testset)
        confusion = get_confusion_matrix(results['predictions'])
        print(confusion['cancel_order']['track_order'])  # Misclassified count
    """
    confusion: dict[str, dict[str, int]] = {}

    for pred in predictions:
        expected = pred["expected"]
        predicted = pred["predicted"]

        if expected not in confusion:
            confusion[expected] = {}

        if predicted not in confusion[expected]:
            confusion[expected][predicted] = 0

        confusion[expected][predicted] += 1

    return confusion


def print_confusion_matrix(predictions: list[dict], show_correct: bool = False) -> None:
    """
    Print a human-readable confusion matrix showing classification errors.

    Args:
        predictions: List of prediction dicts from evaluate()
        show_correct: If True, also show correct predictions (default: only errors)

    Example output:
        CONFUSION MATRIX (Errors Only)
        ----------------------------------------
        Expected: get_refund
          → track_refund: 2 times
          → complaint: 1 time

        Expected: complaint
          → contact_customer_service: 1 time
    """
    confusion = get_confusion_matrix(predictions)

    print("\n" + "=" * 60)
    if show_correct:
        print("CONFUSION MATRIX (All Predictions)")
    else:
        print("CONFUSION MATRIX (Errors Only)")
    print("=" * 60)

    has_errors = False

    # Sort by expected intent for consistent output
    for expected in sorted(confusion.keys()):
        predictions_for_class = confusion[expected]

        # Collect entries to show
        entries_to_show = []
        for predicted, count in sorted(predictions_for_class.items(), key=lambda x: -x[1]):
            is_correct = expected == predicted
            if show_correct or not is_correct:
                entries_to_show.append((predicted, count, is_correct))

        # Only print if there's something to show
        if entries_to_show:
            has_errors = True
            print(f"\nExpected: {expected}")
            for predicted, count, is_correct in entries_to_show:
                marker = "✓" if is_correct else "→"
                times = "time" if count == 1 else "times"
                print(f"  {marker} {predicted}: {count} {times}")

    if not has_errors:
        print("\nNo errors! All predictions were correct.")

    print("=" * 60)


def get_error_summary(predictions: list[dict]) -> list[tuple[str, str, int]]:
    """
    Get a summary of the most common errors.

    Args:
        predictions: List of prediction dicts from evaluate()

    Returns:
        List of (expected, predicted, count) tuples, sorted by count descending

    Example:
        errors = get_error_summary(results['predictions'])
        for expected, predicted, count in errors[:5]:
            print(f"{expected} confused with {predicted}: {count} times")
    """
    confusion = get_confusion_matrix(predictions)
    errors = []

    for expected, predictions_dict in confusion.items():
        for predicted, count in predictions_dict.items():
            if expected != predicted:  # Only errors
                errors.append((expected, predicted, count))

    # Sort by count descending
    errors.sort(key=lambda x: -x[2])
    return errors


# Quick test when run directly
if __name__ == "__main__":
    from dotenv import load_dotenv

    # Add parent directories to path for imports
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_intent_classification_data  # type: ignore[import-not-found]

    # Load environment and configure DSPy
    load_dotenv()
    lm = dspy.LM("openai/gpt-3.5-turbo")
    dspy.configure(lm=lm)

    print("=" * 60)
    print("Intent Classification Module Test")
    print("=" * 60)

    # Load small dataset for testing
    # Note: Need at least 27 samples (one per intent) for stratified sampling
    print("\nLoading data (100 train, 50 test)...")
    trainset, testset = load_intent_classification_data(n_train=100, n_test=50)
    print(f"Train: {len(trainset)}, Test: {len(testset)}")

    # Create and evaluate baseline classifier
    print("\nCreating baseline classifier...")
    classifier = create_classifier()

    print("\nEvaluating baseline...")
    results = evaluate(classifier, testset, verbose=True)

    # Print confusion matrix (errors only)
    print_confusion_matrix(results["predictions"])

    print("\n" + "=" * 60)
    print(f"Final Accuracy: {results['accuracy']:.1f}%")
    print("=" * 60)
