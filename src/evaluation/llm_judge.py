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


class HelpfulnessJudge(dspy.Signature):
    """Judge if the response is helpful and actionable."""

    query: str = dspy.InputField(desc="Original customer query")
    intent: str = dspy.InputField(desc="Customer intent category")
    response: str = dspy.InputField(desc="Generated support response")

    is_helpful: bool = dspy.OutputField(
        desc="True if the response provides useful, actionable information"
    )


class ProfessionalismJudge(dspy.Signature):
    """Judge if the response is professional and appropriate."""

    response: str = dspy.InputField(desc="Generated support response")

    is_professional: bool = dspy.OutputField(
        desc="True if the response uses appropriate, professional language"
    )


class EmpathyJudge(dspy.Signature):
    """Judge if the response shows empathy for the customer's situation."""

    query: str = dspy.InputField(desc="Original customer query")
    response: str = dspy.InputField(desc="Generated support response")

    shows_empathy: bool = dspy.OutputField(
        desc="True if the response acknowledges the customer's concern empathetically"
    )


class CompletenessJudge(dspy.Signature):
    """Judge if the response fully addresses the customer's needs."""

    query: str = dspy.InputField(desc="Original customer query")
    intent: str = dspy.InputField(desc="Customer intent category")
    response: str = dspy.InputField(desc="Generated support response")

    completeness_score: float = dspy.OutputField(
        desc="Score from 0.0 to 1.0 indicating how completely the response addresses the query"
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
# Multi-Dimensional Judge Module
# =============================================================================


class MultiDimensionalJudge(dspy.Module):
    """Combines multiple judge dimensions for comprehensive evaluation."""

    def __init__(self, judge_lm: Optional[dspy.LM] = None):
        super().__init__()
        self.helpfulness = dspy.ChainOfThought(HelpfulnessJudge)
        self.professionalism = dspy.ChainOfThought(ProfessionalismJudge)
        self.empathy = dspy.ChainOfThought(EmpathyJudge)
        self.completeness = dspy.ChainOfThought(CompletenessJudge)
        self.judge_lm = judge_lm

    def forward(self, query: str, intent: str, response: str) -> dspy.Prediction:
        context = dspy.context(lm=self.judge_lm) if self.judge_lm else nullcontext()

        with context:
            h = self.helpfulness(query=query, intent=intent, response=response)
            p = self.professionalism(response=response)
            e = self.empathy(query=query, response=response)
            c = self.completeness(query=query, intent=intent, response=response)

        score = (
            0.40 * float(h.is_helpful)
            + 0.20 * float(p.is_professional)
            + 0.20 * float(e.shows_empathy)
            + 0.20 * c.completeness_score
        )

        return dspy.Prediction(
            quality_score=score,
            helpfulness=h.is_helpful,
            professionalism=p.is_professional,
            empathy=e.shows_empathy,
            completeness=c.completeness_score,
        )


class EvaluationJudge(dspy.Module):
    """Full evaluation judge with all dimensions and reasoning."""

    def __init__(self, judge_lm: Optional[dspy.LM] = None):
        super().__init__()
        self.quality = dspy.ChainOfThought(ResponseQualityJudge)
        self.helpfulness = dspy.ChainOfThought(HelpfulnessJudge)
        self.professionalism = dspy.ChainOfThought(ProfessionalismJudge)
        self.empathy = dspy.ChainOfThought(EmpathyJudge)
        self.completeness = dspy.ChainOfThought(CompletenessJudge)
        self.judge_lm = judge_lm

    def forward(self, query: str, intent: str, response: str) -> dspy.Prediction:
        context = dspy.context(lm=self.judge_lm) if self.judge_lm else nullcontext()

        with context:
            q = self.quality(query=query, intent=intent, response=response)
            h = self.helpfulness(query=query, intent=intent, response=response)
            p = self.professionalism(response=response)
            e = self.empathy(query=query, response=response)
            c = self.completeness(query=query, intent=intent, response=response)

        return dspy.Prediction(
            quality_score=q.quality_score,
            helpfulness=float(h.is_helpful),
            professionalism=float(p.is_professional),
            empathy=float(e.shows_empathy),
            completeness=c.completeness_score,
            reasoning=q.reasoning,
        )


# =============================================================================
# Metric Functions (for DSPy Optimizers)
# =============================================================================

_evaluation_judge = None


def get_judge() -> EvaluationJudge:
    """Get the full evaluation judge with all dimensions and reasoning."""
    global _evaluation_judge
    if _evaluation_judge is None:
        _evaluation_judge = EvaluationJudge()
    return _evaluation_judge


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


_multi_judge = None


def get_multi_judge():
    """Lazy initialization of multi-dimensional judge."""
    global _multi_judge
    if _multi_judge is None:
        _multi_judge = MultiDimensionalJudge()
    return _multi_judge


def multi_dimensional_metric(
    example: dspy.Example, pred: dspy.Prediction, trace=None
) -> float:
    """Multi-dimensional quality metric combining helpfulness, professionalism, empathy, and completeness."""
    judge = get_multi_judge()
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
