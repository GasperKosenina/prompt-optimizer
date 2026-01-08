"""Response Generation Evaluator"""

import dspy
from src.evaluation.llm_judge import get_judge, ComparativeJudge


def evaluate(
    generator: dspy.Module,
    testset: list[dspy.Example],
    verbose: bool = True,
) -> dict:
    """Evaluate a response generator using strict principle-based judges."""
    judge = get_judge()
    all_scores = []

    for i, example in enumerate(testset):
        prediction = generator(query=example.query, intent=example.intent)
        judgment = judge(query=example.query, intent=example.intent, response=prediction.response)

        scores = {
            "quality_score": judgment.quality_score,
            "helpfulness": judgment.helpfulness,
            "completeness": judgment.completeness,
            "empathy": judgment.empathy,
            "professionalism": judgment.professionalism,
            "reasoning": judgment.reasoning,
            "query": example.query,
            "intent": example.intent,
            "response": prediction.response,
        }
        all_scores.append(scores)

        if verbose and i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"Query: {example.query[:80]}...")
            print(f"Intent: {example.intent}")
            print(f"Quality: {scores['quality_score']:.2f}")
            print(f"  Helpfulness:    {scores['helpfulness']:.2f}")
            print(f"  Completeness:   {scores['completeness']:.2f}")
            print(f"  Empathy:        {scores['empathy']:.2f}")
            print(f"  Professionalism:{scores['professionalism']:.2f}")

        if verbose and (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(testset)}...")

    avg = lambda key: sum(s[key] for s in all_scores) / len(all_scores)

    result = {
        "average_quality": avg("quality_score"),
        "dimensions": {
            "helpfulness": avg("helpfulness"),
            "completeness": avg("completeness"),
            "empathy": avg("empathy"),
            "professionalism": avg("professionalism"),
        },
        "scores": all_scores,
        "n": len(testset),
    }

    if verbose:
        print("\n=== Results ===")
        print(f"Quality: {result['average_quality']:.2f}")
        for dim, val in result["dimensions"].items():
            print(f"  {dim}: {val:.2f}")

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

        results["comparisons"].append({
            "query": example.query,
            "winner": judgment.winner,
            "reasoning": judgment.reasoning,
        })

        if verbose and i < 3:
            print(f"\n--- Comparison {i+1} ---")
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
