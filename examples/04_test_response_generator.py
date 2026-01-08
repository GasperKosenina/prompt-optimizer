"""Test response generator with strict principle-based evaluation"""

from dotenv import load_dotenv
import dspy

from src.data.loader import load_query_data
from src.modules.response_generator import create_response_generator, get_optimized_instructions
from src.evaluation.response_evaluator import evaluate

load_dotenv()

lm = dspy.LM("openai/gpt-3.5-turbo")
dspy.configure(lm=lm)

# Load data WITHOUT gold responses
print("Loading data...")
trainset, testset = load_query_data()
print(f"Loaded: {len(trainset)} train, {len(testset)} test")
print("Using first 5 test examples for quick test\n")

# Create baseline generator
print("Creating baseline generator...")
baseline = create_response_generator()

print("Baseline instructions:")
print(f"  '{get_optimized_instructions(baseline)}'")
print()

# Evaluate with strict judges
print("=== Baseline Evaluation (Strict Judges) ===")
results = evaluate(baseline, testset[:5], verbose=True)

# Show generated responses
print("\n=== Generated Responses ===")
for i, score in enumerate(results["scores"], 1):
    print(f"\n--- Response {i} (Quality: {score['quality_score']:.2f}) ---")
    print(f"Query: {score['query']}")
    print(f"Intent: {score['intent']}")
    print(f"\nResponse:\n{score['response']}")
    print(f"\nReasoning: {score['reasoning']}")
