"""Response Generation Evaluator

Simple evaluator using a single strict quality judge.
"""

import dspy
from src.evaluation.llm_judge import ResponseQualityJudge, ComparativeJudge


def evaluate(
    generator: dspy.Module,
    testset: list[dspy.Example],
    verbose: bool = True,
) -> dict:
    """
    Evaluate a response generator using a single strict quality judge.

    Args:
        generator: The response generator module to evaluate
        testset: List of examples with query and intent
        verbose: Print progress and examples

    Returns:
        Dictionary with average_quality, scores list, and statistics
    """
    judge = dspy.ChainOfThought(ResponseQualityJudge)
    all_scores = []

    if verbose:
        print(f"Evaluating {len(testset)} examples...")

    for i, example in enumerate(testset):
        prediction = generator(query=example.query, intent=example.intent)

        judgment = judge(
            query=example.query, intent=example.intent, response=prediction.response
        )

        scores = {
            "quality_score": judgment.quality_score,
            "reasoning": judgment.reasoning,
            "query": example.query,
            "intent": example.intent,
            "response": prediction.response,
        }
        all_scores.append(scores)

        if verbose and i < 3:
            print(f"\n--- Example {i + 1} ---")
            print(f"Query: {example.query[:80]}...")
            print(f"Intent: {example.intent}")
            print(f"Quality: {scores['quality_score']:.2f}")
            print(f"Reasoning: {scores['reasoning'][:100]}...")

        if verbose and (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(testset)} examples evaluated")

    scores_list = [s["quality_score"] for s in all_scores]
    avg_quality = sum(scores_list) / len(scores_list)
    min_quality = min(scores_list)
    max_quality = max(scores_list)

    result = {
        "average_quality": avg_quality,
        "min_quality": min_quality,
        "max_quality": max_quality,
        "scores": all_scores,
        "n": len(testset),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Examples evaluated: {len(testset)}")
        print(f"Average quality:    {avg_quality:.3f}")
        print(f"Min quality:        {min_quality:.3f}")
        print(f"Max quality:        {max_quality:.3f}")
        print(f"Range:              {max_quality - min_quality:.3f}")

        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        print(f"\nScore distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in scores_list if bins[i] <= s < bins[i + 1])
            bar = "█" * count
            print(f"  {bins[i]:.1f}-{bins[i + 1]:.1f}: {bar} ({count})")
        print("=" * 60)

    return result


def compare_responses(
    testset: list[dspy.Example],
    baseline_responses: list[str],
    optimized_responses: list[str],
    verbose: bool = True,
) -> dict:
    """Compare baseline vs optimized responses using comparative judge."""
    judge = dspy.ChainOfThought(ComparativeJudge)
    results = {"baseline_wins": 0, "optimized_wins": 0, "ties": 0, "comparisons": []}

    for i, example in enumerate(testset):
        judgment = judge(
            query=example.query,
            intent=example.intent,
            response_a=baseline_responses[i],
            response_b=optimized_responses[i],
        )

        if judgment.winner == "A":
            results["baseline_wins"] += 1
        elif judgment.winner == "B":
            results["optimized_wins"] += 1
        else:
            results["ties"] += 1

        results["comparisons"].append(
            {
                "query": example.query,
                "winner": judgment.winner,
                "reasoning": judgment.reasoning,
            }
        )

        if verbose and i < 3:
            print(f"\n--- Comparison {i + 1} ---")
            print(f"Query: {example.query[:60]}...")
            print(f"Winner: {judgment.winner}")
            print(f"Reason: {judgment.reasoning}")

    if verbose:
        total = len(testset)
        print("\n=== Comparison Results ===")
        print(f"Baseline wins:  {results['baseline_wins']}/{total}")
        print(f"Optimized wins: {results['optimized_wins']}/{total}")
        print(f"Ties:           {results['ties']}/{total}")

    return results
