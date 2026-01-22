"""
First DSPy Program - Hello DSPy!

This script demonstrates the basic DSPy workflow:
1. Configure the language model
2. Define a signature (input/output specification)
3. Create a predictor
4. Run inference
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)


# Define a simple signature - what the model should do
class AnswerQuestion(dspy.Signature):
    """Answer questions concisely."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def main():
    predictor = dspy.Predict(AnswerQuestion)

    test_questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What color is the sky?",
    ]

    print("=" * 50)
    print("DSPy Hello World - Simple Q&A")
    print("=" * 50)

    for question in test_questions:
        result = predictor(question=question)
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")

    print("\n" + "=" * 50)
    print("Success! DSPy is working correctly.")
    print("=" * 50)


if __name__ == "__main__":
    main()
