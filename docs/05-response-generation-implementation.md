# Response Generation Implementation Guide (Phase 2)

**Quick reference for implementing Part 2 of the research**

---

## Overview

This guide provides concrete implementation steps for the response generation evaluation using **LLM-as-Judge** for quality assessment.

### Key Change from Original Plan

Instead of rule-based metrics (regex patterns, sentence-transformers), we use **LLM-as-Judge** because:
- More nuanced evaluation of tone, empathy, and helpfulness
- Native DSPy integration (same paradigm for generation and evaluation)
- Captures aspects that rules can't (e.g., "Is this response professional?")
- Simpler implementation, no external dependencies

---

## Understanding System Prompts in DSPy

Before implementing, it's crucial to understand **what gets optimized**.

### Where is the "System Prompt"?

In DSPy, there's no explicit system prompt you write. Instead, DSPy generates it from your **Signature**:

```python
class ResponseGenerator(dspy.Signature):
    """Generate customer support response"""  # ← Becomes instructions

    query: str = dspy.InputField(desc="Customer support query")
    intent: str = dspy.InputField(desc="Detected customer intent")
    response: str = dspy.OutputField(desc="Professional, empathetic response")
```

The docstring and field descriptions become the "system prompt" that DSPy sends to the LLM.

### What MIPROv2 Optimizes

MIPROv2 modifies **two components**:

| Component | Location | What Changes |
|-----------|----------|--------------|
| **Instructions** | `signature.instructions` | The task description |
| **Demos** | Few-shot examples | Selected from trainset |

**Before optimization:**
```
"Generate customer support response"
```

**After MIPROv2 optimization:**
```
"You are an expert customer service representative. When responding
to customer queries, always: (1) acknowledge their concern with empathy,
(2) provide clear, actionable steps, (3) use a warm but professional tone,
(4) offer follow-up assistance. Reference order numbers using {{placeholders}}."
```

### How to Inspect Optimized Instructions

```python
# After optimization
optimized = mipro.compile(module, trainset=trainset)

# View the optimized "system prompt"
for name, pred in optimized.named_predictors():
    print(f"=== {name} ===")
    print(pred.signature.instructions)

# Or for simple modules
print(optimized.predict.signature.instructions)

# See full prompt history
dspy.inspect_history()
```

---

## Architecture

```
Query + Intent → ResponseGenerator → Response → LLM Judge → Score
                        ↑                            ↓
                   MIPROv2 optimizes          Uses score to
                   instructions               select best
                   based on scores            instruction
```

---

## Step 1: Data Loader Extension

### Modify `src/data/loader.py`

The function `load_response_generation_data()` already exists. Verify it returns:

```python
def load_response_generation_data(
    n_train: int = 200,
    n_test: int = 100,
    include_intent: bool = True,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Load data for response generation task.

    Returns examples with:
    - query: Customer query
    - intent: Classified intent (if include_intent=True)
    - response: Gold standard response
    """
```

---

## Step 2: Response Generator Module

### Create `src/modules/response_generator.py`

```python
"""
Response Generator Module

Generates customer support responses using DSPy.
The system prompt (instructions) is optimized by MIPROv2.
"""

import dspy


class ResponseGenerator(dspy.Signature):
    """Generate a helpful customer support response."""

    query: str = dspy.InputField(
        desc="Customer support query or message"
    )
    intent: str = dspy.InputField(
        desc="Detected customer intent category (e.g., cancel_order, track_order)"
    )
    response: str = dspy.OutputField(
        desc="Professional, empathetic support response with clear next steps"
    )


def create_response_generator() -> dspy.ChainOfThought:
    """
    Create a response generator with chain-of-thought reasoning.

    ChainOfThought adds a 'reasoning' step before generating the response,
    which often improves quality.
    """
    return dspy.ChainOfThought(ResponseGenerator)


def get_optimized_instructions(module: dspy.Module) -> str:
    """
    Extract the optimized instructions (system prompt) from a compiled module.

    Use this to inspect what MIPROv2 discovered.
    """
    for name, pred in module.named_predictors():
        return pred.signature.instructions
    return ""


def print_all_instructions(module: dspy.Module) -> None:
    """Print instructions for all predictors in a module."""
    for name, pred in module.named_predictors():
        print(f"=== {name} ===")
        print(pred.signature.instructions)
        print()
```

---

## Step 3: LLM-as-Judge Implementation

### Create `src/evaluation/llm_judge.py`

```python
"""
LLM-as-Judge for Response Quality Evaluation

Uses a separate LLM to judge response quality, providing the optimization
signal for MIPROv2. This is more nuanced than rule-based metrics.
"""

import dspy
from typing import Optional


# =============================================================================
# Judge Signatures
# =============================================================================

class ResponseQualityJudge(dspy.Signature):
    """
    Judge the overall quality of a customer support response.
    Consider helpfulness, professionalism, empathy, and completeness.
    """

    query: str = dspy.InputField(desc="Original customer query")
    intent: str = dspy.InputField(desc="Customer intent category")
    response: str = dspy.InputField(desc="Generated support response to evaluate")

    quality_score: float = dspy.OutputField(
        desc="Quality score from 0.0 to 1.0, where 1.0 is excellent"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the score (2-3 sentences)"
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


# =============================================================================
# Judge Module (combines multiple judges)
# =============================================================================

class MultiDimensionalJudge(dspy.Module):
    """
    Combines multiple judge dimensions for comprehensive evaluation.

    Dimensions:
    - Helpfulness (40%): Does it help the customer?
    - Professionalism (20%): Is it appropriate?
    - Empathy (20%): Does it acknowledge concerns?
    - Completeness (20%): Does it fully address the query?
    """

    def __init__(self, judge_lm: Optional[dspy.LM] = None):
        super().__init__()
        self.helpfulness = dspy.ChainOfThought(HelpfulnessJudge)
        self.professionalism = dspy.ChainOfThought(ProfessionalismJudge)
        self.empathy = dspy.ChainOfThought(EmpathyJudge)
        self.completeness = dspy.ChainOfThought(CompletenessJudge)
        self.judge_lm = judge_lm

    def forward(self, query: str, intent: str, response: str) -> dspy.Prediction:
        # Use judge LM if specified (recommended: use different model than generator)
        context = dspy.context(lm=self.judge_lm) if self.judge_lm else nullcontext()

        with context:
            h = self.helpfulness(query=query, intent=intent, response=response)
            p = self.professionalism(response=response)
            e = self.empathy(query=query, response=response)
            c = self.completeness(query=query, intent=intent, response=response)

        # Weighted combination
        score = (
            0.40 * float(h.is_helpful) +
            0.20 * float(p.is_professional) +
            0.20 * float(e.shows_empathy) +
            0.20 * c.completeness_score
        )

        return dspy.Prediction(
            quality_score=score,
            helpfulness=h.is_helpful,
            professionalism=p.is_professional,
            empathy=e.shows_empathy,
            completeness=c.completeness_score,
        )


# =============================================================================
# Metric Functions (for DSPy optimizers)
# =============================================================================

# Simple single-judge approach
_simple_judge = None

def get_simple_judge():
    """Lazy initialization of simple judge."""
    global _simple_judge
    if _simple_judge is None:
        _simple_judge = dspy.ChainOfThought(ResponseQualityJudge)
    return _simple_judge


def response_quality_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Simple quality metric using a single LLM judge.

    Returns: Quality score from 0.0 to 1.0
    """
    judge = get_simple_judge()
    result = judge(
        query=example.query,
        intent=example.intent,
        response=pred.response
    )
    return result.quality_score


# Multi-dimensional judge approach
_multi_judge = None

def get_multi_judge():
    """Lazy initialization of multi-dimensional judge."""
    global _multi_judge
    if _multi_judge is None:
        _multi_judge = MultiDimensionalJudge()
    return _multi_judge


def multi_dimensional_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Multi-dimensional quality metric combining helpfulness, professionalism,
    empathy, and completeness.

    Returns: Weighted quality score from 0.0 to 1.0
    """
    judge = get_multi_judge()
    result = judge(
        query=example.query,
        intent=example.intent,
        response=pred.response
    )
    return result.quality_score


# =============================================================================
# Helper: nullcontext for Python < 3.10 compatibility
# =============================================================================

from contextlib import nullcontext
```

---

## Step 4: Evaluation Runner

### Create `src/evaluation/response_evaluator.py`

```python
"""
Response Generation Evaluator

Evaluates response generators using LLM-as-Judge metrics.
"""

import dspy
from typing import Optional
from src.evaluation.llm_judge import (
    get_simple_judge,
    get_multi_judge,
    ResponseQualityJudge,
)


def evaluate_response_generator(
    generator: dspy.Module,
    testset: list[dspy.Example],
    use_multi_dimensional: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a response generator on a test set using LLM-as-Judge.

    Args:
        generator: The response generator module to evaluate
        testset: List of DSPy Examples with query, intent, response
        use_multi_dimensional: Use multi-judge (True) or simple judge (False)
        verbose: Print progress and examples

    Returns:
        Dictionary with:
        - average_quality: Overall quality score (0.0-1.0)
        - individual_scores: Per-example breakdown
        - dimension_averages: Average for each judge dimension (if multi)
    """
    judge = get_multi_judge() if use_multi_dimensional else get_simple_judge()
    all_scores = []

    for i, example in enumerate(testset):
        # Generate response
        prediction = generator(query=example.query, intent=example.intent)

        # Judge quality
        if use_multi_dimensional:
            judgment = judge(
                query=example.query,
                intent=example.intent,
                response=prediction.response
            )
            scores = {
                "quality_score": judgment.quality_score,
                "helpfulness": judgment.helpfulness,
                "professionalism": judgment.professionalism,
                "empathy": judgment.empathy,
                "completeness": judgment.completeness,
            }
        else:
            judgment = judge(
                query=example.query,
                intent=example.intent,
                response=prediction.response
            )
            scores = {
                "quality_score": judgment.quality_score,
                "reasoning": judgment.reasoning,
            }

        # Add metadata
        scores["query"] = example.query
        scores["intent"] = example.intent
        scores["generated"] = prediction.response
        scores["gold"] = example.response

        all_scores.append(scores)

        if verbose and i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"Query: {example.query[:80]}...")
            print(f"Intent: {example.intent}")
            print(f"Quality: {scores['quality_score']:.3f}")
            if use_multi_dimensional:
                print(f"  Helpful: {scores['helpfulness']}")
                print(f"  Professional: {scores['professionalism']}")
                print(f"  Empathetic: {scores['empathy']}")
                print(f"  Complete: {scores['completeness']:.2f}")

        if verbose and (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(testset)} examples...")

    # Compute averages
    avg_quality = sum(s["quality_score"] for s in all_scores) / len(all_scores)

    result = {
        "average_quality": avg_quality,
        "individual_scores": all_scores,
        "total_examples": len(testset),
    }

    if use_multi_dimensional:
        result["dimension_averages"] = {
            "helpfulness": sum(s["helpfulness"] for s in all_scores) / len(all_scores),
            "professionalism": sum(s["professionalism"] for s in all_scores) / len(all_scores),
            "empathy": sum(s["empathy"] for s in all_scores) / len(all_scores),
            "completeness": sum(s["completeness"] for s in all_scores) / len(all_scores),
        }

    if verbose:
        print(f"\n=== Final Results ===")
        print(f"Average Quality: {avg_quality:.3f}")
        if use_multi_dimensional:
            print("Dimension Averages:")
            for dim, val in result["dimension_averages"].items():
                print(f"  {dim}: {val:.3f}")

    return result
```

---

## Step 5: Running the Optimization Pipeline

### Usage Example

```python
from dotenv import load_dotenv
import dspy

from src.data.loader import load_response_generation_data
from src.modules.response_generator import (
    ResponseGenerator,
    create_response_generator,
    get_optimized_instructions,
)
from src.evaluation.llm_judge import response_quality_metric

# Setup
load_dotenv()

# Use different models for generation vs judging to avoid bias
generator_lm = dspy.LM("openai/gpt-3.5-turbo")
judge_lm = dspy.LM("openai/gpt-4o-mini")  # Stronger model as judge

dspy.configure(lm=generator_lm)

# Load data
trainset, testset = load_response_generation_data(n_train=200, n_test=50)
print(f"Loaded: {len(trainset)} train, {len(testset)} test")

# Create baseline generator
baseline = create_response_generator()

# Evaluate baseline
print("\n=== Baseline Evaluation ===")
from src.evaluation.response_evaluator import evaluate_response_generator
baseline_results = evaluate_response_generator(baseline, testset[:20])

# Run MIPROv2 optimization
print("\n=== Running MIPROv2 Optimization ===")
optimizer = dspy.MIPROv2(
    metric=response_quality_metric,
    auto="light",
    num_threads=4,
)

optimized = optimizer.compile(
    baseline.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)

# Inspect what MIPROv2 discovered
print("\n=== Optimized Instructions ===")
print(get_optimized_instructions(optimized))

# Evaluate optimized generator
print("\n=== Optimized Evaluation ===")
optimized_results = evaluate_response_generator(optimized, testset[:20])

# Compare
print("\n=== Comparison ===")
print(f"Baseline Quality:  {baseline_results['average_quality']:.3f}")
print(f"Optimized Quality: {optimized_results['average_quality']:.3f}")
print(f"Improvement: {optimized_results['average_quality'] - baseline_results['average_quality']:+.3f}")

# Save optimized module
optimized.save("results/response_generator_optimized.json")
```

---

## Step 6: Key Research Questions

With LLM-as-Judge, you can now investigate:

### 1. What Instructions Does MIPROv2 Generate?

```python
# After optimization
print("=== Discovered System Prompt ===")
print(get_optimized_instructions(optimized))
```

Compare instructions across:
- Different optimization runs (stability)
- Different judge configurations (what matters)
- Bootstrap vs MIPROv2 (optimizer differences)

### 2. Which Instruction Components Correlate with Quality?

Analyze the optimized instructions for common patterns:
- Empathy language ("acknowledge", "understand")
- Structure guidance ("numbered steps", "clear")
- Tone instructions ("professional", "warm")
- Constraints ("under 150 words", "no promises")

### 3. How Do Different Judges Affect Optimization?

Try different judge configurations:
```python
# Judge A: Prioritize helpfulness
weights_helpful = {"helpfulness": 0.6, "professionalism": 0.2, ...}

# Judge B: Prioritize empathy
weights_empathy = {"empathy": 0.5, "helpfulness": 0.3, ...}

# Compare resulting instructions
```

---

## Cost Considerations

### LLM-as-Judge Adds API Calls

During optimization, every evaluation requires judge calls:
- MIPROv2 `auto="light"`: ~50-100 evaluations
- Each evaluation: 1 generation + 1-4 judge calls
- Total: ~200-500 API calls per optimization run

### Mitigation Strategies

1. **Use cheap judge model**: `gpt-4o-mini` or `claude-3-haiku`
2. **Smaller eval batches**: Reduce trainset size during development
3. **Cache judge responses**: Same response → same score
4. **Simple judge first**: Single judge vs multi-dimensional

### Cost Estimate

| Model | Role | Cost/1K tokens | Calls | Est. Cost |
|-------|------|----------------|-------|-----------|
| GPT-3.5 | Generator | $0.0005 | 200 | ~$0.10 |
| GPT-4o-mini | Judge | $0.00015 | 800 | ~$0.12 |
| **Total** | | | | **~$0.25/run** |

---

## Troubleshooting

### Issue: Judge scores are always high (>0.9)

**Cause**: Judge is too lenient or signature is vague.

**Solution**: Make judge criteria more specific:
```python
quality_score: float = dspy.OutputField(
    desc="Score from 0.0 to 1.0. Give 0.8+ only if response is excellent. "
         "Average responses should score 0.5-0.7. Poor responses below 0.4."
)
```

### Issue: Scores vary wildly for same response

**Cause**: LLM non-determinism.

**Solution**:
- Set temperature=0 for judge
- Average multiple judge calls
- Use more specific criteria

### Issue: Optimization doesn't improve scores

**Solutions**:
1. Increase training set (200 → 300 examples)
2. Try `auto="medium"` instead of `auto="light"`
3. Check if judge is properly differentiating quality
4. Verify judge uses different LM than generator

### Issue: Judge and generator use same model (circular)

**Problem**: Model might judge its own outputs favorably.

**Solution**: Always use different models:
```python
generator_lm = dspy.LM("openai/gpt-3.5-turbo")
judge_lm = dspy.LM("openai/gpt-4o-mini")
# or
judge_lm = dspy.LM("anthropic/claude-3-haiku")
```

---

## Success Criteria

This implementation will be successful if:

| Metric | Target |
|--------|--------|
| Baseline quality | 0.45 - 0.60 |
| Optimized quality | 0.70 - 0.85 |
| Improvement | > 0.15 (15+ points) |
| GPT-4 baseline | 0.80 - 0.90 |
| Optimized GPT-3.5 vs GPT-4 | ≥ 80% |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/modules/response_generator.py` | ResponseGenerator signature |
| `src/evaluation/llm_judge.py` | LLM-as-Judge signatures and metrics |
| `src/evaluation/response_evaluator.py` | Evaluation runner |
| `results/response_generator_optimized.json` | Saved optimized module |

---

## Next Steps

1. **Implement** the modules above
2. **Test** with small dataset (n=20) to verify pipeline works
3. **Run** baseline evaluation
4. **Optimize** with MIPROv2
5. **Analyze** the discovered instructions
6. **Compare** with GPT-4 baseline
7. **Document** findings for research output

---

**Document Version**: 2.0
**Last Updated**: January 2026
**Major Change**: Switched from rule-based metrics to LLM-as-Judge
