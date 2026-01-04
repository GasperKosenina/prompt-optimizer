"""
Intent Classification with DSPy

This script demonstrates classifying customer support queries into intent categories.
This is Task 1 from the project overview - a good starting point because:
- Objective evaluation (exact match accuracy)
- Fast and cheap to run
- Clear success criteria
"""

import dspy
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Configure DSPy with GPT-3.5-turbo (cost-effective)
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


class IntentClassifier(dspy.Signature):
    """Classify customer support query into an intent category."""

    query: str = dspy.InputField(desc="Customer support query")
    intent: str = dspy.OutputField(
        desc="Intent category: REFUND, ORDER_STATUS, COMPLAINT, RETURN, BILLING, SHIPPING, PRODUCT_INFO, or OTHER"
    )


def main():
    # Create predictor
    classifier = dspy.Predict(IntentClassifier)

    # Test queries covering different intents
    test_queries = [
        # REFUND
        "I want my money back for this broken product",
        "Can I get a refund for my order?",
        # ORDER_STATUS
        "Where is my package?",
        "When will my order arrive?",
        # COMPLAINT
        "Your service is terrible, I'm very upset",
        "This is the worst experience I've ever had",
        # RETURN
        "How do I return this item?",
        "I need to send this product back",
        # BILLING
        "I was charged twice for my order",
        "There's an error on my invoice",
        # SHIPPING
        "Do you ship to Canada?",
        "What are your shipping options?",
        # PRODUCT_INFO
        "What sizes does this come in?",
        "Is this product compatible with iPhone?",
    ]

    print("=" * 60)
    print("Intent Classification Demo")
    print("=" * 60)

    for query in test_queries:
        result = classifier(query=query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result.intent}")

    print("\n" + "=" * 60)
    print("Classification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
