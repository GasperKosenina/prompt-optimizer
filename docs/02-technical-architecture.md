# Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT RUNNER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Dataset    │───▶│  DSPy Module │───▶│  Evaluator   │      │
│  │   Loader     │    │  (Signature) │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   ▼                   │               │
│         │           ┌──────────────┐            │               │
│         │           │  Optimizer   │            │               │
│         │           │ (MIPRO/Boot) │            │               │
│         │           └──────────────┘            │               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────┐       │
│  │                   Results Logger                     │       │
│  │          (metrics, costs, optimized prompts)         │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## DSPy Signatures

### Task 1: Intent Classification

```python
class IntentClassifier(dspy.Signature):
    """Classify customer support query into intent category."""

    query: str = dspy.InputField(desc="Customer support query")
    intent: str = dspy.OutputField(desc="Intent category")
```

### Task 2: Response Generation

```python
class ResponseGenerator(dspy.Signature):
    """Generate helpful customer support response."""

    query: str = dspy.InputField(desc="Customer support query")
    intent: str = dspy.InputField(desc="Classified intent category")  # optional
    response: str = dspy.OutputField(desc="Helpful support response")
```

## Optimizers to Compare

### 1. BootstrapFewShot (Baseline Optimizer)
- **How it works**: Automatically selects few-shot examples from training data
- **Cost**: Low (few LLM calls for bootstrapping)
- **Expected improvement**: Moderate

### 2. MIPRO v2 (Advanced Optimizer)
- **How it works**: Uses meta-prompting to generate and evaluate prompt variations
- **Cost**: High (many LLM calls for meta-optimization)
- **Expected improvement**: Higher

### 3. COPRO (Optional)
- **How it works**: Coordinate-wise prompt optimization
- **Cost**: Medium
- **Expected improvement**: TBD

## LLM Backends

| Model | Provider | Use Case | Cost Level |
|-------|----------|----------|------------|
| `gpt-3.5-turbo` | OpenAI | Budget inference | Low |
| `gpt-4o-mini` | OpenAI | Better quality, still affordable | Medium |
| `claude-3-haiku` | Anthropic | Fast, cheap | Low |
| `llama3.1:8b` | Ollama (local) | Development, free | Free |

## Evaluation Pipeline

### Intent Classification Metrics
```python
def evaluate_classification(predictions, ground_truth):
    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "f1_macro": f1_score(ground_truth, predictions, average='macro'),
        "f1_weighted": f1_score(ground_truth, predictions, average='weighted'),
    }
```

### Response Generation Metrics

1. **Semantic Similarity** (cheap, automated)
   - Compute embeddings for generated and ground truth responses
   - Calculate cosine similarity
   - Models: `text-embedding-3-small` or local `sentence-transformers`

2. **LLM-as-Judge** (more expensive, use on sample)
   - Prompt an LLM to rate response quality (1-5 scale)
   - Evaluate: helpfulness, accuracy, tone
   - Apply to random sample to control costs

## Dataset Strategy

### Development Phase
- **Size**: 100-200 examples
- **Selection**: Stratified sample (equal representation of intents)
- **Purpose**: Fast iteration, debug pipelines

### Evaluation Phase
- **Size**: 500-1000 examples (budget permitting)
- **Selection**: Stratified, separate from training
- **Purpose**: Final metrics for thesis

### Dataset Splits
```
Full Dataset (27K)
    │
    ├── Development Set (200)
    │       ├── Train (140)
    │       └── Test (60)
    │
    └── Evaluation Set (1000) [held out for final experiments]
            ├── Train (700)
            └── Test (300)
```

## Cost Tracking

Every experiment will log:
- Total API calls
- Input/output tokens
- Estimated cost (USD)
- Wall-clock time

```python
@dataclass
class ExperimentCost:
    api_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    duration_seconds: float
```

## Project Structure (Proposed)

```
gasper-prompt-optimizer/
├── docs/
│   ├── 01-project-overview.md
│   ├── 02-technical-architecture.md
│   └── 03-experiment-results.md
├── src/
│   ├── data/
│   │   ├── loader.py          # Dataset loading & preprocessing
│   │   └── sampler.py         # Stratified sampling
│   ├── modules/
│   │   ├── intent_classifier.py
│   │   └── response_generator.py
│   ├── optimizers/
│   │   └── runner.py          # Optimizer experiment runner
│   ├── evaluation/
│   │   ├── classification.py  # Accuracy, F1
│   │   ├── semantic.py        # Embedding similarity
│   │   └── llm_judge.py       # LLM-as-judge
│   └── utils/
│       ├── cost_tracker.py
│       └── logger.py
├── experiments/
│   └── configs/               # Experiment configurations
├── results/
│   └── [experiment outputs]
├── notebooks/
│   └── analysis.ipynb         # Results visualization
└── tests/
```

---

*Document created: 2025-01-03*
*Status: Draft - Technical Planning*
