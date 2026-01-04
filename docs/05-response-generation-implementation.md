# Response Generation Implementation Guide

**Quick reference for implementing Part 2 of the research**

---

## Overview

This guide provides concrete implementation steps for the response generation evaluation, the core contribution of this research.

---

## Architecture

```
Query → Classify Intent → Generate Response → Evaluate Quality → Compare Results
   ↓           ↓                  ↓                  ↓                 ↓
Bitext     IntentClassifier   ResponseGenerator   QualityMetrics   Visualizations
           (Part 1 ✓)         (Part 2 TODO)       (Part 2 TODO)    (Part 2 TODO)
```

---

## Step 1: Data Loader Extension

### Modify `src/data/loader.py`

Add function to load query + response pairs:

```python
def load_response_generation_data(
    n_train: int = 200,
    n_test: int = 100,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Load data for response generation task.

    Returns examples with:
    - query: Customer query
    - intent: Classified intent (optional, can include for context)
    - response: Gold standard response
    """
    df = load_bitext_dataset()

    # Create examples with query + response
    examples = []
    for _, row in df.iterrows():
        examples.append(
            dspy.Example(
                query=row["instruction"],
                intent=row["intent"],
                response=row["response"]
            ).with_inputs("query", "intent")  # Only query+intent are inputs
        )

    # Stratified split
    trainset, testset = split_dataset(
        examples,
        n_train=n_train,
        n_test=n_test,
        stratify_by="intent"
    )

    return trainset, testset
```

---

## Step 2: Response Generator Module

### Create `src/modules/response_generator.py`

```python
import dspy

class ResponseGenerator(dspy.Signature):
    """Generate customer support response"""

    query: str = dspy.InputField(
        desc="Customer support query or message"
    )
    intent: str = dspy.InputField(
        desc="Detected customer intent (e.g., cancel_order, track_order)"
    )

    response: str = dspy.OutputField(
        desc="Professional, empathetic support response with clear steps"
    )


def create_response_generator() -> dspy.ChainOfThought:
    """Create a response generator with chain-of-thought reasoning."""
    return dspy.ChainOfThought(ResponseGenerator)
```

**Why include intent?**
- Gives the model context about what type of response is needed
- Matches real-world scenario (you'd classify first, then generate)
- Allows evaluation of "given correct intent, how good is the response?"

---

## Step 3: Quality Metrics Implementation

### Create `src/evaluation/quality_metrics.py`

```python
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Load semantic similarity model once
_similarity_model = None

def get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _similarity_model


def semantic_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts (0-1)."""
    model = get_similarity_model()
    embeddings = model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def has_acknowledgment(response: str) -> bool:
    """Check if response acknowledges customer concern."""
    acknowledgment_patterns = [
        r"\b(understand|sorry|apologize|appreciate|thank)\b",
        r"\b(I see|I get|I hear)\b",
        r"\b(frustrat|concern|issue|problem)\b",
    ]
    text_lower = response.lower()
    return any(re.search(pattern, text_lower) for pattern in acknowledgment_patterns)


def has_actionable_steps(response: str) -> bool:
    """Check if response provides clear steps or information."""
    # Look for numbered steps, bullet points, or instructional language
    step_patterns = [
        r"\b\d+\.",  # Numbered list (1., 2., 3.)
        r"^[\-\*]",  # Bullet points (-, *)
        r"\b(follow these steps|here's how|you can|please)\b",
        r"\b(visit|click|navigate|go to|contact)\b",
    ]
    return any(re.search(pattern, response, re.MULTILINE) for pattern in step_patterns)


def appropriate_length(response: str, min_words: int = 30, max_words: int = 250) -> bool:
    """Check if response length is appropriate."""
    word_count = len(response.split())
    return min_words <= word_count <= max_words


def uses_placeholders(response: str) -> bool:
    """Check if response uses company placeholders like {{Order Number}}."""
    return bool(re.search(r"\{\{[^\}]+\}\}", response))


def offers_followup(response: str) -> bool:
    """Check if response offers further assistance."""
    followup_patterns = [
        r"\b(if you|should you|feel free|don't hesitate)\b",
        r"\b(further|additional|more) (help|assistance|questions)\b",
        r"\b(here to help|happy to assist|reach out)\b",
    ]
    text_lower = response.lower()
    return any(re.search(pattern, text_lower) for pattern in followup_patterns)


def compute_quality_score(
    generated_response: str,
    gold_response: str,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive quality score.

    Args:
        generated_response: Model-generated response
        gold_response: Gold standard response from dataset
        weights: Optional custom weights for metrics

    Returns:
        Dictionary with individual scores and composite quality score
    """
    if weights is None:
        weights = {
            "semantic_similarity": 0.30,
            "has_acknowledgment": 0.15,
            "has_actionable_steps": 0.20,
            "appropriate_length": 0.10,
            "uses_placeholders": 0.10,
            "offers_followup": 0.15,
        }

    # Compute individual metrics
    scores = {
        "semantic_similarity": semantic_similarity(generated_response, gold_response),
        "has_acknowledgment": float(has_acknowledgment(generated_response)),
        "has_actionable_steps": float(has_actionable_steps(generated_response)),
        "appropriate_length": float(appropriate_length(generated_response)),
        "uses_placeholders": float(uses_placeholders(generated_response)),
        "offers_followup": float(offers_followup(generated_response)),
    }

    # Compute weighted composite score
    composite_score = sum(scores[k] * weights[k] for k in weights)
    scores["composite_quality"] = composite_score

    return scores


def response_quality_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """
    DSPy-compatible metric function for response generation.

    Returns composite quality score (0.0-1.0).
    """
    scores = compute_quality_score(
        generated_response=prediction.response,
        gold_response=example.response,
    )
    return scores["composite_quality"]
```

---

## Step 4: Evaluation Runner

### Create `src/evaluation/response_evaluator.py`

```python
import dspy
from src.evaluation.quality_metrics import compute_quality_score

def evaluate_response_generator(
    generator: dspy.Module,
    testset: list[dspy.Example],
    verbose: bool = True,
) -> dict:
    """
    Evaluate response generator on test set.

    Returns:
        Dictionary with:
        - average_quality: Overall quality score
        - individual_scores: Per-example breakdown
        - metric_averages: Average for each metric dimension
    """
    all_scores = []

    for i, example in enumerate(testset):
        # Generate response
        prediction = generator(query=example.query, intent=example.intent)

        # Compute quality
        scores = compute_quality_score(
            generated_response=prediction.response,
            gold_response=example.response,
        )

        scores["query"] = example.query
        scores["intent"] = example.intent
        scores["generated"] = prediction.response
        scores["gold"] = example.response

        all_scores.append(scores)

        if verbose and i < 5:  # Print first 5 examples
            print(f"\n--- Example {i+1} ---")
            print(f"Query: {example.query[:100]}...")
            print(f"Quality: {scores['composite_quality']:.3f}")

    # Compute averages
    metric_averages = {}
    for key in ["semantic_similarity", "has_acknowledgment", "has_actionable_steps",
                "appropriate_length", "uses_placeholders", "offers_followup", "composite_quality"]:
        metric_averages[key] = sum(s[key] for s in all_scores) / len(all_scores)

    return {
        "average_quality": metric_averages["composite_quality"],
        "individual_scores": all_scores,
        "metric_averages": metric_averages,
    }
```

---

## Step 5: Running the Pipeline

### Usage Example

```python
from dotenv import load_dotenv
import dspy
from src.data.loader import load_response_generation_data
from src.modules.response_generator import ResponseGenerator
from src.evaluation.quality_metrics import response_quality_metric
from src.optimizers.runner import run_mipro_v2

# Setup
load_dotenv()
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

# Load data
trainset, testset = load_response_generation_data(n_train=200, n_test=100)

# Run optimization
optimized_generator, result = run_mipro_v2(
    trainset=trainset,
    testset=testset,
    classifier_class=ResponseGenerator,
    metric_fn=response_quality_metric,
    auto="light",
)

# Results
print(f"Baseline Quality: {result.baseline_accuracy:.1f}%")
print(f"Optimized Quality: {result.optimized_accuracy:.1f}%")
print(f"Improvement: {result.improvement:+.1f}%")
```

---

## Step 6: Visualization

### Add to `src/optimizers/runner.py`

Create response-specific plots:
- Quality score breakdown (semantic, structural, tone)
- Example comparisons (baseline vs optimized vs GPT-4)
- Cost-quality tradeoff curve

---

## Expected Workflow

```bash
# 1. Run classification (already done)
python -m src.optimizers.runner -o both -t 100 -e 50

# 2. Run response generation baseline
python -m src.evaluation.response_evaluator --baseline

# 3. Run response generation optimization
python -m src.evaluation.response_evaluator --optimize

# 4. Compare with GPT-4
python -m src.evaluation.response_evaluator --gpt4-baseline

# 5. Generate comparison report
python -m src.evaluation.generate_report
```

---

## Key Implementation Notes

### 1. Optimization Target
Unlike classification (which optimizes for accuracy), response generation optimizes for **composite quality score** (multi-dimensional).

### 2. Metric Weights
You can adjust metric weights based on business priorities:
```python
# Example: Prioritize accuracy over formatting
weights = {
    "semantic_similarity": 0.50,  # Increased
    "has_acknowledgment": 0.10,
    "has_actionable_steps": 0.20,
    "appropriate_length": 0.05,  # Decreased
    "uses_placeholders": 0.05,   # Decreased
    "offers_followup": 0.10,
}
```

### 3. Human Evaluation
For validation, sample 100 examples and collect human ratings:
- Use Google Forms or custom web interface
- Show responses without labels (blind evaluation)
- Ask: "Which response would you prefer?" + ratings

### 4. GPT-4 Comparison
To establish upper bound:
```python
lm_gpt4 = dspy.LM("openai/gpt-4")
with dspy.context(lm=lm_gpt4):
    gpt4_generator = dspy.ChainOfThought(ResponseGenerator)
    gpt4_results = evaluate_response_generator(gpt4_generator, testset)
```

---

## Troubleshooting

### Issue: Semantic similarity always high (>0.9)
**Solution**: Model may be copying gold responses. Check that training doesn't include test set.

### Issue: Quality scores too low (<0.3)
**Solution**:
- Try GPT-4 baseline to see if it's a model limitation
- Check if prompts are too restrictive
- Validate quality metrics aren't too harsh

### Issue: MIPROv2 optimization doesn't improve
**Solution**:
- Increase training set size (200 → 300)
- Try `auto="medium"` instead of `auto="light"`
- Check metric function is working correctly

---

## Success Criteria

This implementation will be successful if:
- ✅ Baseline quality score: 0.50-0.60
- ✅ Optimized quality score: 0.70-0.80
- ✅ GPT-4 quality score: 0.80-0.90
- ✅ Clear improvement over baseline (>0.15 absolute gain)

---

## Next Steps

1. Implement data loader extension
2. Create ResponseGenerator module
3. Implement quality metrics
4. Run baseline evaluation
5. Run MIPROv2 optimization
6. Compare results and visualize

**Estimated time**: 2-3 days of focused work

---

**Document Version**: 1.0
**Last Updated**: January 4, 2026
