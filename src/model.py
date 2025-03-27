from typing import Optional
import numpy as np
from src.tree import DecisionTree
from src.utils import compute_lambdas, evaluate_ndcg


class LambdaMART:
    """
    The LambdaMART class implements a LambdaMART model for learning to rank. It uses gradient-boosted 
    decision trees, where each tree is trained to predict the lambdas (pseudo-gradients) that indicate 
    how to improve the ranking. The model is trained iteratively, with each new tree correcting the
    errors of the previous trees. The predict method combines the predictions of all the trees, and the
    score method evaluates the model's performance using the Normalized Discounted Cumulative Gain (NDCG.
    This class relies on the DecisionTree class (in tree.py) for building individual trees and the 
    compute_lambdas and evaluate_ndcg functions (in utils.py) for calculating lambdas and NDCG, respectively.

    Parameters:
    -----------
    n_estimators : int
        Number of trees in the ensemble to build.

    learning_rate : float
        Controls the contribution of each tree to the final prediction. Smaller values mean slower 
        learning and more regularization.

    max_depth : int
        Maximum depth of the each tree, preventing overfitting.

    min_samples_split : int
        Minimum number of samples required to split an internal node in a tree.

    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 random_state: Optional[int] = None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []  # List of trees in the ensemble
        self.scores = None  # Scores for each sample, which will be updated during training
        self.base_scores = None  # Base scores initialized during training

    def fit(self, x: np.ndarray, y: np.ndarray, query_ids: np.ndarray):
        """
        Train the LambdaMART model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (relevance scores).

        query_ids : array-like of shape (n_samples,)
            Query identifiers for each sample.
        """
        # Initialize each sample's score with random values around the mean
        np.random.seed(self.random_state)
        self.base_scores = np.random.normal(
            loc=np.mean(y), scale=0.1, size=x.shape[0])
        self.scores = self.base_scores.copy()

        # "Boosting Loop" - Train the ensemble of trees. Each iteration adds one new tree by
        # correcting the previous predictions.
        for _ in range(self.n_estimators):
            # Compute lambdas and weights from the current scores, true relevance (y), and query ids.
            # lambdas are the pseudo-gradients that indicate how much the score of an item should change
            # to improve its ranking relative to other items.
            # weights are the relative importance of each item in the ranking.
            lambdas, weights = compute_lambdas(
                self.scores, y, query_ids
            )

            # Train a decision tree on the lambdas and weights
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            tree.fit(x, lambdas, weights)

            # Update scores for the next iteration
            self.scores += self.learning_rate * tree.predict(x)

            # Store the tree
            self.trees.append(tree)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores for x using the trained model.

        Parameters:
        -----------
        x : array-like of shape (n_samples, n_features)
            The input samples. Each sample has n_features.

        Returns:
        --------
        array, shape = [n_samples]
            The predicted relevance scores. The higher the score, the more relevant the item.
        """
        if not hasattr(self, 'base_scores'):
            raise RuntimeError(
                "Model must be trained before making predictions")

        # Create base scores for the test set using the same mean as training
        test_base_scores = np.random.normal(
            loc=np.mean(self.base_scores),
            scale=0.1,
            size=x.shape[0]
        )

        predictions = np.zeros(x.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(x)

        # Normalize the scores to be between 0 and 1
        min_score = np.min(test_base_scores + predictions)
        max_score = np.max(test_base_scores + predictions)
        normalized_scores = (test_base_scores + predictions -
                             min_score) / (max_score - min_score)

        return normalized_scores

    def score(self, x: np.ndarray, y: np.ndarray, query_ids: np.ndarray) -> float:
        """
        Return the NDCG score of the predictions.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The test input samples.

        y : array-like of shape (n_samples,)
            The true relevance scores.

        query_ids : array-like of shape (n_samples,)
            Query identifiers for each sample.

        Returns:
        --------
        float
            The NDCG score.
        """
        predictions = self.predict(x)
        return evaluate_ndcg(y, predictions, query_ids)
