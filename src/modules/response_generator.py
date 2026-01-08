"""Response Generator Module"""

import dspy


class ResponseGenerator(dspy.Signature):
    """Generate a helpful customer support response."""

    query: str = dspy.InputField(desc="Customer support query or message")
    intent: str = dspy.InputField(desc="Detected customer intent (e.g., cancel_order, track_order)")
    response: str = dspy.OutputField(desc="Professional, empathetic support response with clear next steps")


def create_response_generator() -> dspy.ChainOfThought:
    """Create a response generator with chain-of-thought reasoning."""
    return dspy.ChainOfThought(ResponseGenerator)


def get_optimized_instructions(module: dspy.Module) -> str:
    """Extract the optimized instructions (system prompt) from a compiled module."""
    for _, pred in module.named_predictors():
        return getattr(pred.signature, "instructions", "")
    return ""


def print_all_instructions(module: dspy.Module) -> None:
    """Print instructions for all predictors in a module."""
    for name, pred in module.named_predictors():
        print(f"=== {name} ===")
        print(getattr(pred.signature, "instructions", ""))
        print()
