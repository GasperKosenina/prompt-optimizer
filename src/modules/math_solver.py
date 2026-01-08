"""
Math Word Problem Solver Module

Provides utilities for solving grade-school math problems from GSM8K dataset.
"""

import re
import dspy


def extract_answer(answer_text: str) -> str:
    """
    Extract the final numerical answer from GSM8K format.

    GSM8K answers look like: "reasoning steps\n#### 42"
    We extract just the number after ####

    Args:
        answer_text: Full answer string from GSM8K

    Returns:
        Just the final number as a string

    Example:
        >>> extract_answer("Step 1: 3+5=8\\n#### 8")
        "8"
    """
    if "####" in answer_text:
        return answer_text.split("####")[1].strip()
    return answer_text.strip()


def parse_number(text: str) -> float | None:
    """
    Parse a number from text, handling common variations.

    Handles: "42", "42.0", "$42", "42 dollars", etc.

    Args:
        text: String potentially containing a number

    Returns:
        The number as float, or None if no number found

    Example:
        >>> parse_number("$42.50")
        42.5
        >>> parse_number("42 apples")
        42.0
    """
    # Remove common non-numeric characters
    text = text.replace("$", "").replace(",", "").strip()

    # Extract first number (including decimals and negatives)
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def answers_match(expected: str, predicted: str) -> bool:
    """
    Check if two answers match (with tolerance for formatting differences).

    Handles variations like:
    - "42" vs "42.0"
    - "42" vs "$42"
    - "42" vs "42 dollars"

    Args:
        expected: Expected answer (ground truth)
        predicted: Predicted answer

    Returns:
        True if answers match, False otherwise

    Example:
        >>> answers_match("42", "42.0")
        True
        >>> answers_match("42", "$42")
        True
        >>> answers_match("42", "43")
        False
    """
    # Try numeric comparison first
    exp_num = parse_number(expected)
    pred_num = parse_number(predicted)

    if exp_num is not None and pred_num is not None:
        # Allow small tolerance for rounding (0.01)
        return abs(exp_num - pred_num) < 0.01

    # Fallback to string comparison
    return expected.strip().lower() == predicted.strip().lower()


# =============================================================================
# DSPy Signature and Module
# =============================================================================


class MathSolver(dspy.Signature):
    """Solve math word problems using step-by-step reasoning."""

    question: str = dspy.InputField(
        desc="A math word problem requiring arithmetic reasoning"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning showing how to solve the problem"
    )
    answer: str = dspy.OutputField(desc="The final numerical answer (just the number)")


def create_math_solver() -> dspy.ChainOfThought:
    """
    Create a math solver with chain-of-thought reasoning.

    ChainOfThought is essential for math problems - it adds explicit
    reasoning steps which significantly improve accuracy.

    Returns:
        A DSPy ChainOfThought module configured with MathSolver signature

    Example:
        >>> solver = create_math_solver()
        >>> result = solver(question="Sally has 3 apples. Tom gives her 5 more. How many total?")
        >>> print(result.answer)
        "8"
    """
    return dspy.ChainOfThought(MathSolver)


def math_accuracy_metric(
    example: dspy.Example, pred: dspy.Prediction, trace=None
) -> bool:
    """
    Metric function for math problem accuracy (used by DSPy optimizers).

    Returns True if the predicted answer matches the expected answer.

    Args:
        example: A DSPy Example with 'answer' field (ground truth)
        pred: A DSPy Prediction with 'answer' field (model output)
        trace: Optional trace (unused, required by DSPy)

    Returns:
        True if answers match, False otherwise

    Example:
        >>> example = dspy.Example(question="2+2?", answer="4")
        >>> pred = solver(question="2+2?")
        >>> math_accuracy_metric(example, pred)
        True
    """
    expected = extract_answer(example.answer)
    predicted = pred.answer
    return answers_match(expected, predicted)
