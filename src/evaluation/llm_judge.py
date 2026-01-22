"""LLM-as-Judge for Response Quality Evaluation"""

import dspy
from contextlib import nullcontext
from typing import Optional


# =============================================================================
# Judge Signatures
# =============================================================================


class ResponseQualityJudge(dspy.Signature):
    """Judge the overall quality of a customer support response with STRICT standards.

    Be critical and demanding! Most responses should score in the 0.5-0.7 range.
    Only truly exceptional responses that go above and beyond deserve 0.8+.
    """

    query: str = dspy.InputField(desc="Original customer query")
    intent: str = dspy.InputField(desc="Customer intent category")
    response: str = dspy.InputField(desc="Generated support response to evaluate")

    quality_score: float = dspy.OutputField(
        desc="""Quality score from 0.0 to 1.0 using STRICT standards:
        
        - 0.8-1.0: EXCELLENT - Exceptional response that goes above and beyond. 
                   Shows deep empathy, provides specific actionable steps, anticipates 
                   follow-up needs, uses perfect professional tone.
        
        - 0.6-0.8: GOOD - Solid response with minor room for improvement. Addresses 
                   the query adequately but could be more specific, empathetic, or complete.
        
        - 0.4-0.6: ACCEPTABLE - Basic response that addresses the query but missing 
                   important elements like empathy, specificity, or next steps.
        
        - 0.2-0.4: POOR - Unhelpful or incomplete response. May be too generic, 
                   dismissive, or fails to properly address the customer's needs.
        
        - 0.0-0.2: VERY POOR - Inappropriate, rude, or completely fails to address 
                   the query.
        
        BE CRITICAL! Most baseline responses should score 0.5-0.7."""
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the score focusing on what's missing or could be improved (2-3 sentences)"
    )


class ComparativeJudge(dspy.Signature):
    """Compare two responses and determine which is better."""

    query: str = dspy.InputField(desc="Original customer query")
    intent: str = dspy.InputField(desc="Customer intent category")
    response_a: str = dspy.InputField(desc="First response (baseline)")
    response_b: str = dspy.InputField(desc="Second response (optimized)")

    winner: str = dspy.OutputField(desc="Which response is better: 'A', 'B', or 'TIE'")
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this response is better"
    )


# =============================================================================
# Metric Functions (for DSPy Optimizers)
# =============================================================================


_simple_judge = None


def get_simple_judge():
    """Lazy initialization of simple judge."""
    global _simple_judge
    if _simple_judge is None:
        _simple_judge = dspy.ChainOfThought(ResponseQualityJudge)
    return _simple_judge


def response_quality_metric(
    example: dspy.Example, pred: dspy.Prediction, trace=None
) -> float:
    """Simple quality metric using a single LLM judge."""
    judge = get_simple_judge()
    result = judge(query=example.query, intent=example.intent, response=pred.response)
    return result.quality_score


def strict_quality_metric(
    example: dspy.Example, pred: dspy.Prediction, trace=None
) -> float:
    """
    Simple strict quality metric using a single judge.

    This is the recommended metric for optimization - simpler, faster, and easier to calibrate.
    Returns a score from 0.0 to 1.0 where most baseline responses score 0.5-0.7.
    """
    judge = get_simple_judge()
    result = judge(query=example.query, intent=example.intent, response=pred.response)
    return result.quality_score
