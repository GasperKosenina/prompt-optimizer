# Quick Start Guide (Beginner-Friendly)

This guide will get you from zero to running DSPy experiments.

## Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install dspy-ai
pip install pandas
pip install scikit-learn
pip install sentence-transformers  # for semantic similarity
pip install python-dotenv  # for API keys
```

## Step 2: API Keys

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=your-openai-key-here
# Optional:
ANTHROPIC_API_KEY=your-anthropic-key-here
```

**Important:** Add `.env` to `.gitignore` to avoid committing secrets!

## Step 3: Your First DSPy Program

```python
# examples/01_hello_dspy.py

import dspy
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Configure DSPy with a language model
lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

# Define a simple signature (what the model should do)
class AnswerQuestion(dspy.Signature):
    """Answer questions concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Create a predictor
predictor = dspy.Predict(AnswerQuestion)

# Use it!
result = predictor(question="What is the capital of France?")
print(result.answer)
```

Run it:
```bash
python examples/01_hello_dspy.py
```

## Step 4: Intent Classification Example

```python
# examples/02_intent_classifier.py

import dspy
from dotenv import load_dotenv

load_dotenv()
dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))

# Define signature for intent classification
class IntentClassifier(dspy.Signature):
    """Classify customer support query into an intent category."""

    query: str = dspy.InputField(desc="Customer support query")
    intent: str = dspy.OutputField(desc="Intent category like REFUND, ORDER_STATUS, COMPLAINT, etc.")

# Create predictor
classifier = dspy.Predict(IntentClassifier)

# Test it
test_queries = [
    "I want my money back for this broken product",
    "Where is my package?",
    "Your service is terrible, I'm very upset",
]

for query in test_queries:
    result = classifier(query=query)
    print(f"Query: {query}")
    print(f"Intent: {result.intent}")
    print("---")
```

## Step 5: Optimization with BootstrapFewShot

```python
# examples/03_optimize_classifier.py

import dspy
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

load_dotenv()
dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))

# Signature
class IntentClassifier(dspy.Signature):
    """Classify customer support query into an intent category."""
    query: str = dspy.InputField()
    intent: str = dspy.OutputField()

# Create training data as dspy.Example objects
trainset = [
    dspy.Example(query="I want a refund", intent="REFUND").with_inputs("query"),
    dspy.Example(query="Where is my order?", intent="ORDER_STATUS").with_inputs("query"),
    dspy.Example(query="This product is defective", intent="COMPLAINT").with_inputs("query"),
    dspy.Example(query="How do I return this?", intent="RETURN").with_inputs("query"),
    # Add more examples...
]

# Define a metric function
def accuracy_metric(example, prediction, trace=None):
    """Check if predicted intent matches expected intent."""
    return example.intent.lower() == prediction.intent.lower()

# Create optimizer
optimizer = BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=4,  # Max few-shot examples to generate
    max_labeled_demos=4,       # Max labeled examples to include
    max_rounds=1,              # Keep low for budget
)

# Create base program
classifier = dspy.Predict(IntentClassifier)

# Compile (optimize) the program
print("Optimizing...")
optimized_classifier = optimizer.compile(
    student=classifier,
    trainset=trainset,
)

# Save for later use
optimized_classifier.save("optimized_classifier.json")
print("Saved optimized classifier!")

# Test the optimized version
result = optimized_classifier(query="I need my money back")
print(f"Predicted intent: {result.intent}")
```

## Step 6: Evaluation

```python
# examples/04_evaluate.py

import dspy
from dotenv import load_dotenv

load_dotenv()
dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))

# Load optimized classifier
class IntentClassifier(dspy.Signature):
    query: str = dspy.InputField()
    intent: str = dspy.OutputField()

optimized_classifier = dspy.Predict(IntentClassifier)
optimized_classifier.load("optimized_classifier.json")

# Test set (separate from training!)
testset = [
    dspy.Example(query="Give me a refund now", intent="REFUND").with_inputs("query"),
    dspy.Example(query="Track my package", intent="ORDER_STATUS").with_inputs("query"),
    # Add more...
]

# Metric
def accuracy_metric(example, prediction, trace=None):
    return example.intent.lower() == prediction.intent.lower()

# Evaluate
evaluator = dspy.Evaluate(
    devset=testset,
    metric=accuracy_metric,
    display_progress=True,
    display_table=5,
)

score = evaluator(optimized_classifier)
print(f"Accuracy: {score}%")
```

## Common Patterns

### Loading the Bitext Dataset

```python
import pandas as pd

# Download from GitHub and load
df = pd.read_csv("path/to/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Convert to DSPy Examples
def row_to_example(row):
    return dspy.Example(
        query=row['instruction'],  # or whatever the column name is
        intent=row['intent'],
        response=row['response'],
    ).with_inputs("query")

examples = [row_to_example(row) for _, row in df.iterrows()]
```

### Stratified Sampling

```python
from sklearn.model_selection import train_test_split

# Stratified split to keep intent distribution balanced
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['intent'],  # Keep intent proportions
    random_state=42,
)
```

### Cost-Conscious Tips

1. **Use cheaper models for development:**
   ```python
   dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))  # Cheap
   # dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # Slightly more expensive
   ```

2. **Start with tiny datasets:**
   ```python
   trainset = trainset[:50]  # Only 50 examples for testing
   ```

3. **Use MIPRO "light" mode:**
   ```python
   optimizer = MIPROv2(metric=metric, auto="light")  # Not "heavy"!
   ```

4. **Set low max_rounds:**
   ```python
   optimizer = BootstrapFewShot(metric=metric, max_rounds=1)
   ```

---

## Next Steps

1. Run the examples above to verify your setup works
2. Load the actual Bitext dataset
3. Start with classification task
4. Add response generation
5. Compare optimizers

---

*Document created: 2025-01-03*
