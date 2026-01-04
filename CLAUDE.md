# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DSPy-based prompt optimization research project focused on customer support intent classification and response generation. The project uses the Bitext Customer Support Training Dataset (27K examples, 27 intent categories) to evaluate different DSPy optimizers (BootstrapFewShot and MIPROv2).

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys (copy .env.example to .env and fill in values)
cp .env.example .env
```

### Running Examples
```bash
# Basic DSPy hello world
python examples/01_hello_dspy.py

# Intent classification demo
python examples/02_intent_classifier.py

# Optimize classifier with BootstrapFewShot
python examples/03_optimize_classifier.py
```

### Running Modules Directly
```bash
# Test data loader
python -m src.data.loader

# Test intent classifier with evaluation
python -m src.modules.intent_classifier

# Run optimizer comparison
python -m src.optimizers.runner --optimizer bootstrap --train-size 100 --test-size 50
python -m src.optimizers.runner --optimizer mipro --mipro-auto light
python -m src.optimizers.runner --optimizer both
```

## Architecture

### Core Pipeline
The project follows a modular DSPy optimization pipeline:

1. **Data Loading** (`src/data/loader.py`): Loads Bitext dataset, performs stratified sampling, converts to DSPy Examples
2. **DSPy Modules** (`src/modules/`): Defines task signatures (IntentClassifier, ResponseGenerator)
3. **Optimizers** (`src/optimizers/runner.py`): Runs BootstrapFewShot or MIPROv2 optimization
4. **Evaluation** (`src/modules/intent_classifier.py`): Computes accuracy metrics and confusion matrices

### Data Flow
```
Bitext CSV → load_bitext_dataset() → stratified sampling →
create_intent_examples() → DSPy Examples →
optimizer.compile() → optimized classifier →
evaluate() → metrics + results
```

### DSPy Signatures

All DSPy signatures are defined in `src/modules/`. The main signature is:

```python
class IntentClassifier(dspy.Signature):
    query: str = dspy.InputField(desc="Customer support query")
    intent: str = dspy.OutputField(desc="One of 27 intent categories")
```

The 27 intent labels are defined in `src/modules/intent_classifier.py` as `INTENT_LABELS`.

### Optimization Strategy

Two optimizers are implemented in `src/optimizers/runner.py`:

- **BootstrapFewShot**: Fast, low-cost optimizer that selects good few-shot examples from training data
- **MIPROv2**: Advanced optimizer using meta-prompting to generate and evaluate prompt variations (higher cost, potentially better results)

Both return an `OptimizationResult` dataclass with baseline accuracy, optimized accuracy, improvement, duration, and optimizer-specific settings.

### Dataset Handling

The data loader (`src/data/loader.py`) provides several convenience functions:

- `load_intent_classification_data(n_train, n_test)`: Loads and splits data for intent classification
- `get_stratified_sample()`: Ensures proportional representation of each intent class
- `split_dataset()`: Performs train/test split with stratification

Dataset location: `datasets/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

### Cost Considerations

The project is designed for budget-conscious experimentation:
- Default LM is `gpt-3.5-turbo` (cheap)
- Small dataset samples for development (100-200 examples)
- BootstrapFewShot uses `max_rounds=1` by default
- MIPROv2 defaults to `auto="light"` mode

Results are saved to `results/` directory as JSON files.

## Key Implementation Details

### Metric Function
All optimizers use `accuracy_metric()` from `src/modules/intent_classifier.py`:
```python
def accuracy_metric(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    return example.intent.lower().strip() == prediction.intent.lower().strip()
```

### Evaluation
The `evaluate()` function in `src/modules/intent_classifier.py` returns:
- Overall accuracy percentage
- Correct/total counts
- Per-example predictions with correctness flags

Additional utilities:
- `print_confusion_matrix()`: Shows classification errors
- `get_error_summary()`: Lists most common misclassifications

### Example Workflow
1. Load data: `trainset, testset = load_intent_classification_data(n_train=200, n_test=50)`
2. Run optimizer: `optimized, result = run_bootstrap_fewshot(trainset, testset, IntentClassifier, accuracy_metric)`
3. Save: `optimized.save("results/optimized.json")` and `save_results({"bootstrap": result}, "results/metrics.json")`
4. Analyze: Review metrics, confusion matrix, error patterns

## Research Methodology

This project investigates **when prompt optimization matters**:

- **Part 1 (Classification)**: Shows modern LLMs already perform well (~90% accuracy), yielding minimal improvement (+2-4%) from optimization
- **Part 2 (Response Generation)**: Complex tasks show significant improvement (>15%) from optimized system prompts

See `docs/03-research-methodology.md` for full research design and `docs/05-response-generation-implementation.md` for implementation guide.

**Key Insight**: Not all tasks benefit equally from optimization. Simple pattern matching (classification) sees minimal gains, while complex generation tasks (responses) show major improvement.

## File Structure Notes

- `examples/`: Progressive tutorial scripts (01-03) demonstrating DSPy basics through optimization
- `src/data/`: Dataset loading with stratified sampling support
- `src/modules/`: Task-specific DSPy signatures and evaluation logic
- `src/optimizers/`: Experiment runner with optimizer comparison utilities
- `src/evaluation/`: Quality metrics for response generation (Part 2)
- `results/`: Output directory for optimized models, metrics (JSON), and visualizations (PNG)
- `docs/`: Project documentation including research methodology and implementation guides
