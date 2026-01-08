"""
Bitext Dataset Loader

Loads the Bitext Customer Support Training Dataset and converts it to DSPy Examples.
Supports stratified sampling for budget-friendly development.
"""

from pathlib import Path
from typing import cast

import dspy
import pandas as pd
from sklearn.model_selection import train_test_split

# Default path to dataset
DEFAULT_DATASET_PATH = (
    Path(__file__).parent.parent.parent
    / "datasets"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def load_bitext_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the Bitext dataset from CSV.

    Args:
        path: Path to CSV file. If None, uses default location.

    Returns:
        DataFrame with columns: flags, instruction, category, intent, response
    """
    if path is None:
        path = DEFAULT_DATASET_PATH

    df = pd.read_csv(path)
    return df


def create_intent_examples(df: pd.DataFrame) -> list[dspy.Example]:
    """
    Convert DataFrame to DSPy Examples for intent classification.

    Task: instruction -> intent

    Args:
        df: DataFrame with 'instruction' and 'intent' columns

    Returns:
        List of DSPy Examples with query and intent fields
    """
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            query=row["instruction"],
            intent=row["intent"],
        ).with_inputs("query")
        examples.append(example)
    return examples


def create_query_examples(df: pd.DataFrame) -> list[dspy.Example]:
    """
    Convert DataFrame to DSPy Examples with query + intent only (no gold response).

    Use this for principle-based evaluation where we judge quality without
    comparing to gold responses.
    """
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            query=row["instruction"],
            intent=row["intent"],
        ).with_inputs("query", "intent")
        examples.append(example)
    return examples


def create_response_examples(
    df: pd.DataFrame, include_intent: bool = True
) -> list[dspy.Example]:
    """
    Convert DataFrame to DSPy Examples for response generation.

    Task: instruction (+ intent) -> response

    Args:
        df: DataFrame with 'instruction', 'intent', and 'response' columns
        include_intent: If True, include intent as input field

    Returns:
        List of DSPy Examples
    """
    examples = []
    for _, row in df.iterrows():
        if include_intent:
            example = dspy.Example(
                query=row["instruction"],
                intent=row["intent"],
                response=row["response"],
            ).with_inputs("query", "intent")
        else:
            example = dspy.Example(
                query=row["instruction"],
                response=row["response"],
            ).with_inputs("query")
        examples.append(example)
    return examples


def get_stratified_sample(
    df: pd.DataFrame,
    n_samples: int,
    stratify_by: str = "intent",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Get a stratified sample from the dataset.

    Ensures proportional representation of each class.

    Args:
        df: Full DataFrame
        n_samples: Number of samples to return
        stratify_by: Column to stratify by ('intent' or 'category')
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    if n_samples >= len(df):
        return df

    # Use train_test_split to get stratified sample
    # We sample n_samples and discard the rest
    frac = n_samples / len(df)
    sampled_result, _ = train_test_split(
        df,
        train_size=frac,
        stratify=df[stratify_by],
        random_state=random_state,
    )
    sampled = cast(pd.DataFrame, sampled_result)

    # Adjust if we got slightly more or fewer due to rounding
    if len(sampled) > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=random_state)

    return sampled.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_by: str = "intent",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets with stratification.

    Args:
        df: Full DataFrame
        test_size: Fraction for test set (default 0.2 = 20%)
        stratify_by: Column to stratify by
        random_state: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    train_result, test_result = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_by],
        random_state=random_state,
    )
    train_df = cast(pd.DataFrame, train_result)
    test_df = cast(pd.DataFrame, test_result)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_intent_classification_data(
    n_train: int | None = None,
    n_test: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Convenience function to load data for intent classification experiments.

    Args:
        n_train: Max training examples (None = use all)
        n_test: Max test examples (None = use all from split)
        test_size: Fraction for test split
        random_state: Random seed

    Returns:
        Tuple of (trainset, testset) as DSPy Examples
    """
    # Load full dataset
    df = load_bitext_dataset()

    # Split first (before sampling to avoid data leakage)
    train_df, test_df = split_dataset(
        df, test_size=test_size, random_state=random_state
    )

    # Sample if requested
    if n_train is not None:
        train_df = get_stratified_sample(train_df, n_train, random_state=random_state)

    if n_test is not None:
        test_df = get_stratified_sample(test_df, n_test, random_state=random_state)

    # Convert to DSPy Examples
    trainset = create_intent_examples(train_df)
    testset = create_intent_examples(test_df)

    return trainset, testset


def load_response_generation_data(
    n_train: int | None = None,
    n_test: int | None = None,
    include_intent: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Convenience function to load data for response generation experiments.

    Args:
        n_train: Max training examples (None = use all)
        n_test: Max test examples (None = use all from split)
        include_intent: Include intent as input field
        test_size: Fraction for test split
        random_state: Random seed

    Returns:
        Tuple of (trainset, testset) as DSPy Examples
    """
    # Load full dataset
    df = load_bitext_dataset()

    # Split first
    train_df, test_df = split_dataset(
        df, test_size=test_size, random_state=random_state
    )

    # Sample if requested
    if n_train is not None:
        train_df = get_stratified_sample(train_df, n_train, random_state=random_state)

    if n_test is not None:
        test_df = get_stratified_sample(test_df, n_test, random_state=random_state)

    # Convert to DSPy Examples
    trainset = create_response_examples(train_df, include_intent=include_intent)
    testset = create_response_examples(test_df, include_intent=include_intent)

    return trainset, testset


def load_query_data(
    n_train: int | None = None,
    n_test: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Load query + intent data WITHOUT gold responses.

    Use this for principle-based evaluation where we judge quality
    without comparing to gold responses.
    """
    df = load_bitext_dataset()

    train_df, test_df = split_dataset(df, test_size=test_size, random_state=random_state)

    if n_train is not None:
        train_df = get_stratified_sample(train_df, n_train, random_state=random_state)

    if n_test is not None:
        test_df = get_stratified_sample(test_df, n_test, random_state=random_state)

    trainset = create_query_examples(train_df)
    testset = create_query_examples(test_df)

    return trainset, testset


def get_intent_labels() -> list[str]:
    """Return list of all 27 intent labels in the dataset."""
    df = load_bitext_dataset()
    return sorted(df["intent"].unique().tolist())


def get_category_labels() -> list[str]:
    """Return list of all 11 category labels in the dataset."""
    df = load_bitext_dataset()
    return sorted(df["category"].unique().tolist())


# Quick test when run directly
if __name__ == "__main__":
    print("Loading Bitext dataset...")
    df = load_bitext_dataset()
    print(f"Total examples: {len(df)}")
    print(f"Intents: {df['intent'].nunique()}")
    print(f"Categories: {df['category'].nunique()}")

    print("\nLoading intent classification data (200 train, 50 test)...")
    trainset, testset = load_intent_classification_data(n_train=200, n_test=50)
    print(f"Train: {len(trainset)}, Test: {len(testset)}")
    print("\nSample train example:")
    print(f"  Query: {trainset[0].query[:60]}...")
    print(f"  Intent: {trainset[0].intent}")
