from typing import Tuple
import numpy as np


def compute_lambdas(scores: np.ndarray,
                    true_relevance: np.ndarray,
                    query_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lambda values and weights for LambdaMART.
    """
    if len(scores) != len(true_relevance) or len(scores) != len(query_ids):
        raise ValueError("Input arrays must have the same length")

    unique_queries = np.unique(query_ids)
    n_samples = len(scores)
    lambdas = np.zeros(n_samples)
    weights = np.zeros(n_samples)

    for query_id in unique_queries:
        query_mask = query_ids == query_id
        query_scores = scores[query_mask]
        query_true = true_relevance[query_mask]
        query_idx = np.where(query_mask)[0]

        # Compute pairwise differences
        score_diff = query_scores[:, None] - query_scores[None, :]
        relevance_diff = query_true[:, None] - query_true[None, :]

        # Compute ideal DCG
        ideal_dcg = np.sum((2**query_true - 1) /
                           np.log2(np.arange(1, len(query_true) + 1) + 1))

        # Compute DCG differences
        dcg_diff = (2**query_true[:, None] - 1) / \
            np.log2(np.arange(1, len(query_true) + 1) + 2)
        dcg_diff = dcg_diff - \
            (2**query_true[None, :] - 1) / \
            np.log2(np.arange(1, len(query_true) + 1) + 2)

        # Compute lambda values
        lambda_values = np.zeros_like(score_diff)
        # Handle both positive and negative score differences
        lambda_mask_pos = score_diff > 0
        lambda_mask_neg = score_diff < 0
        lambda_values[lambda_mask_pos] = relevance_diff[lambda_mask_pos] * \
            dcg_diff[lambda_mask_pos] / ideal_dcg
        lambda_values[lambda_mask_neg] = relevance_diff[lambda_mask_neg] * \
            dcg_diff[lambda_mask_neg] / ideal_dcg

        # Compute weights
        weight_values = np.zeros_like(score_diff)
        weight_mask = score_diff != 0  # Consider all non-zero score differences
        weight_values[weight_mask] = np.abs(relevance_diff[weight_mask])

        # Update lambdas and weights
        for i, idx_i in enumerate(query_idx):
            for j, _idx_j in enumerate(query_idx):
                if i != j:
                    lambdas[idx_i] += lambda_values[i, j]
                    weights[idx_i] += weight_values[i, j]

    return lambdas, weights


def evaluate_ndcg(true_relevance: np.ndarray,
                  predicted_scores: np.ndarray,
                  query_ids: np.ndarray,
                  k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Parameters:
    -----------
    true_relevance : array-like of shape (n_samples,)
        True relevance scores.

    predicted_scores : array-like of shape (n_samples,)
        Predicted scores.

    query_ids : array-like of shape (n_samples,)
        Query identifiers for each sample.

    k : int
        The position to compute NDCG at.

    Returns:
    --------
    float
        The NDCG score.
    """
    if len(true_relevance) != len(predicted_scores) or len(true_relevance) != len(query_ids):
        raise ValueError("Input arrays must have the same length")
    unique_queries = np.unique(query_ids)
    ndcg_scores = []

    for query_id in unique_queries:
        query_mask = query_ids == query_id
        query_true = true_relevance[query_mask]
        query_pred = predicted_scores[query_mask]

        # Sort by predicted scores
        sorted_idx = np.argsort(query_pred)[::-1]
        sorted_true = query_true[sorted_idx]

        # Compute DCG using min(k, len(sorted_true)) to handle cases with fewer items than k
        effective_k = min(k, len(sorted_true))
        dcg = np.sum((2**sorted_true[:effective_k] - 1) /
                     np.log2(np.arange(1, effective_k + 1) + 1))

        # Compute ideal DCG using the same effective_k
        ideal_dcg = np.sum((2**np.sort(query_true)[::-1][:effective_k] - 1) /
                           np.log2(np.arange(1, effective_k + 1) + 1))

        # Compute NDCG
        if ideal_dcg == 0:
            ndcg_scores.append(0)
        else:
            ndcg_scores.append(dcg / ideal_dcg)

    return np.mean(ndcg_scores)


def create_example_data(n_queries: int = 100,
                        n_features: int = 10,
                        n_samples_per_query: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create example data for testing LambdaMART.

    Parameters:
    -----------
    n_queries : int
        Number of queries to generate.

    n_features : int
        Number of features per sample.

    n_samples_per_query : int
        Number of samples per query.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        x (features), y (relevance scores as integers in range [0,4]), query_ids
    """
    if n_queries <= 0 or n_features <= 0 or n_samples_per_query <= 0:
        raise ValueError("Invalid input values")

    np.random.seed(42)
    total_samples = n_queries * n_samples_per_query

    # Generate features with some structure
    x = np.random.randn(total_samples, n_features)

    # Generate relevance scores based on the features
    # Higher values in the features should correlate with higher relevance
    # Add some noise to make it more realistic
    relevance_base = (x[:, 0] * 2 + x[:, 1] * 1.5).reshape(-1, 1)
    noise = np.random.randn(total_samples, 1) * 0.5
    relevance_scores = relevance_base + noise

    # Convert to discrete relevance scores (0-4)
    y = np.clip(relevance_scores, 0, 4).astype(int).flatten()

    query_ids = np.repeat(np.arange(n_queries), n_samples_per_query)

    return x, y, query_ids
