# Examples Guide

This guide explains the flow of each example script - from baseline evaluation to optimization.

## Overview

The examples demonstrate **DSPy prompt optimization** across three task types:

| Example | Task | Optimizer | Key Finding |
|---------|------|-----------|-------------|
| 01 | Hello World | - | DSPy basics |
| 02-03 | Intent Classification | BootstrapFewShot | LLMs already good (~92%) |
| 04-05 | Response Generation | MIPROv2 | Moderate improvement (+3%) |
| 06-08 | Math Reasoning | MIPROv2 | Large improvement (+20-24%) |

---

## Core Concepts

### DSPy Signature
A signature defines **what** the LLM should do:

```python
class IntentClassifier(dspy.Signature):
    """Classify customer support query into intent."""
    query: str = dspy.InputField()
    intent: str = dspy.OutputField()
```

### DSPy Predictor
A predictor **executes** the signature:

```python
classifier = dspy.Predict(IntentClassifier)
result = classifier(query="I want a refund")
print(result.intent)  # "get_refund"
```

### DSPy Optimizer
An optimizer **improves** the predictor by:
- Finding good few-shot examples (BootstrapFewShot)
- Discovering better instructions (MIPROv2)

---

## Example Flow

### Standard Pattern

Every optimization example follows this pattern:

```
1. Load Data         → trainset, testset
2. Create Baseline   → dspy.Predict(Signature)
3. Evaluate Baseline → accuracy/quality score
4. Run Optimizer     → optimizer.compile(student, trainset)
5. Evaluate Optimized→ compare to baseline
6. Save Results      → JSON files
```

---

## Examples 01-03: Intent Classification

### 01_hello_dspy.py
**Purpose:** Verify DSPy works

```python
# Simple Q&A to test setup
predictor = dspy.Predict(AnswerQuestion)
result = predictor(question="What is 2+2?")
```

### 02_intent_classifier.py
**Purpose:** Establish baseline for classification

```
Flow:
1. Load Bitext dataset (27 intent categories)
2. Create zero-shot classifier
3. Evaluate on 50 test examples
4. Report baseline accuracy
```

### 03_optimize_classifier.py
**Purpose:** Attempt to improve classification with few-shot examples

```
Flow:
1. Load 200 train, 50 test examples
2. Evaluate baseline → 92%
3. Run BootstrapFewShot (finds good examples)
4. Evaluate optimized → 90%
5. Save to results/intent_classification/
```

**Key Insight:** Baseline is already 92%. No room for improvement - LLMs are already good at simple classification.

---

## Examples 04-05: Response Generation

### 04_test_response_generator.py
**Purpose:** Calibrate quality evaluation

```
Flow:
1. Load customer queries with intents
2. Generate responses with baseline
3. Evaluate with LLM-as-Judge
4. Check if baseline scores ~0.65-0.75
```

**Why LLM-as-Judge?** No "correct" answer exists for response generation. We use GPT-4o-mini to score quality (0-1).

### 05_optimize_response_generator.py
**Purpose:** Discover better instructions for responses

```
Flow:
1. Load 100 train, 50 test examples
2. Evaluate baseline → 0.713 quality
3. Run MIPROv2 (discovers new instructions)
4. Evaluate optimized → 0.735 quality
5. Save to results/response_generation/
```

**What MIPROv2 Discovers:**
```
Before: "Generate a helpful customer support response."
After:  "Given a customer support query and the detected customer
        intent, predict the reasoning behind the query and generate
        a tailored response that aligns with both..."
```

---

## Examples 06-08: Math Reasoning

### 06_test_math_solver.py
**Purpose:** Establish math reasoning baseline

```
Flow:
1. Load GSM8K dataset (grade-school math)
2. Create Chain-of-Thought solver
3. Evaluate on 50 problems
4. Report baseline accuracy (~64%)
```

**Chain-of-Thought:** Forces step-by-step reasoning before the final answer.

### 07_optimize_math_solver.py
**Purpose:** Improve math solving with MIPROv2 Light

```
Flow:
1. Load 200 train, 50 test problems
2. Evaluate baseline → 64%
3. Run MIPROv2 Light (~10 min)
4. Evaluate optimized → 84%
5. Save to results/math_solver/light_*.json
```

### 08_advanced_optimization_strategies.py
**Purpose:** Test MIPROv2 Medium with more data

```
Flow:
1. Load 300 train, 50 test problems
2. Evaluate baseline → 64%
3. Run MIPROv2 Medium (~13 min)
4. Evaluate optimized → 88%
5. Save to results/math_solver/medium_*.json
```

---

## Optimizers Explained

### BootstrapFewShot
**How it works:**
1. Run baseline on training examples
2. Collect examples where baseline succeeds
3. Use successful examples as few-shot demonstrations

**Best for:** Simple tasks, fast iteration

### MIPROv2
**How it works:**
1. Generate candidate instructions using meta-prompting
2. Generate candidate few-shot example sets
3. Use Bayesian optimization to find best combination
4. Return optimized predictor

**Modes:**
- `light`: 10 trials, fast (~10 min)
- `medium`: 20 trials, balanced (~30 min)
- `heavy`: 40 trials, thorough (~60 min)

**Best for:** Complex tasks, production use

---

## Metrics

### Classification: Accuracy
```python
def accuracy_metric(example, prediction):
    return example.intent.lower() == prediction.intent.lower()
```

### Response Generation: LLM-as-Judge
```python
class ResponseQualityJudge(dspy.Signature):
    query: str = dspy.InputField()
    response: str = dspy.InputField()
    quality_score: float = dspy.OutputField(desc="0.0 to 1.0")
```

### Math: Answer Matching
```python
def math_accuracy_metric(example, prediction):
    expected = extract_answer(example.answer)  # "#### 42" → 42
    return answers_match(expected, prediction.answer)
```

---

## Results Summary

| Task | Baseline | Optimized | Change |
|------|----------|-----------|--------|
| Intent Classification | 92.0% | 90.0% | -2.0% |
| Response Generation | 0.713 | 0.735 | +3.1% |
| Math Solver (Light) | 64.0% | 84.0% | +20.0% |
| Math Solver (Medium) | 64.0% | 88.0% | +24.0% |

### Key Findings

1. **Simple tasks (classification):** LLMs already perform well. Optimization doesn't help.

2. **Moderate tasks (response generation):** Small gains from better instructions.

3. **Complex tasks (math reasoning):** Large gains from optimization. Few-shot examples and refined instructions significantly improve performance.

---

## Project Structure

```
examples/
├── 01_hello_dspy.py          # DSPy basics
├── 02_intent_classifier.py   # Classification baseline
├── 03_optimize_classifier.py # Classification optimization
├── 04_test_response_generator.py    # Response baseline
├── 05_optimize_response_generator.py # Response optimization
├── 06_test_math_solver.py    # Math baseline
├── 07_optimize_math_solver.py # Math optimization (light)
└── 08_advanced_optimization_strategies.py # Math optimization (medium)

src/
├── data/loader.py            # Dataset loading
├── modules/
│   ├── intent_classifier.py  # Classification signature
│   ├── response_generator.py # Response signature
│   └── math_solver.py        # Math signature
├── evaluation/
│   ├── llm_judge.py          # Quality scoring
│   └── response_evaluator.py # Evaluation runner
└── optimizers/runner.py      # Optimization utilities

results/
├── intent_classification/    # Classification results
├── response_generation/      # Response results
├── math_solver/              # Math results
└── visualizations/           # Charts
```

---

## Running Examples

```bash
# Activate environment
source venv/bin/activate

# Run in order
python examples/01_hello_dspy.py      # ~5 sec
python examples/02_intent_classifier.py # ~1 min
python examples/03_optimize_classifier.py # ~2 min
python examples/04_test_response_generator.py # ~2 min
python examples/05_optimize_response_generator.py # ~10 min
python examples/06_test_math_solver.py # ~2 min
python examples/07_optimize_math_solver.py # ~15 min
python examples/08_advanced_optimization_strategies.py # ~15 min
```

---

## Conclusion

**When does prompt optimization matter?**

- **Not much:** Classification, simple pattern matching
- **Somewhat:** Response generation, moderate complexity
- **A lot:** Math reasoning, multi-step logic

The key insight: **Task complexity determines optimization value.** Simple tasks leave no room for improvement. Complex tasks benefit significantly from discovering better instructions and examples.
