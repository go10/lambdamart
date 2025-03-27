from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    """A node in the decision tree."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None


class DecisionTree:
    """
    Decision tree implementation for LambdaMART.
    """

    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree = None

    def _best_split(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[int, float]:
        """
        Find the best split for the current node.
        """
        best_error = float('inf')
        best_idx = None
        best_threshold = None

        m = len(x)
        if m <= 1:
            return best_idx, best_threshold

        # Calculate total weighted sum
        total_weighted_sum = np.sum(y * weights)
        total_weight = np.sum(weights)

        for idx in range(x.shape[1]):
            # Sort samples by feature value
            sorted_idx = np.argsort(x[:, idx])
            x_sorted = x[sorted_idx, idx]
            y_sorted = y[sorted_idx]
            weights_sorted = weights[sorted_idx]

            weighted_sum = 0
            weight_sum = 0

            for i in range(1, m):
                weighted_sum += y_sorted[i-1] * weights_sorted[i-1]
                weight_sum += weights_sorted[i-1]

                # Calculate weighted MSE
                left_weighted_sum = weighted_sum
                right_weighted_sum = total_weighted_sum - weighted_sum
                left_weight_sum = weight_sum
                right_weight_sum = total_weight - weight_sum

                if left_weight_sum == 0 or right_weight_sum == 0:
                    continue

                left_mean = left_weighted_sum / left_weight_sum
                right_mean = right_weighted_sum / right_weight_sum

                left_error = np.sum(
                    weights_sorted[:i] * (y_sorted[:i] - left_mean) ** 2)
                right_error = np.sum(
                    weights_sorted[i:] * (y_sorted[i:] - right_mean) ** 2)
                error = left_error + right_error

                if error < best_error:
                    best_error = error
                    best_idx = idx
                    best_threshold = (x_sorted[i-1] + x_sorted[i]) / 2

        return best_idx, best_threshold

    def _calculate_node_value(self, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate the weighted average value for a node.
        """
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0
        return np.sum(y * weights) / total_weight

    def _build_tree(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        """
        if depth >= self.max_depth or len(x) < self.min_samples_split:
            return Node(value=self._calculate_node_value(y, weights))

        feature_idx, threshold = self._best_split(x, y, weights)

        if feature_idx is None or threshold is None:
            return Node(value=self._calculate_node_value(y, weights))

        left_idx = x[:, feature_idx] <= threshold
        right_idx = x[:, feature_idx] > threshold

        if np.all(left_idx) or np.all(right_idx):
            return Node(value=self._calculate_node_value(y, weights))

        left_node = self._build_tree(
            x[left_idx], y[left_idx], weights[left_idx], depth + 1)
        right_node = self._build_tree(
            x[right_idx], y[right_idx], weights[right_idx], depth + 1)

        return Node(feature_idx, threshold, left_node, right_node)

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
        """
        Build a decision tree from the training set (x, y).
        """
        self.tree = self._build_tree(x, y, weights)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target values for x.
        """
        def predict_single(x: np.ndarray, node: Node) -> float:
            if node.value is not None:
                return node.value
            if x[node.feature_idx] <= node.threshold:
                return predict_single(x, node.left)
            return predict_single(x, node.right)

        return np.array([predict_single(val, self.tree) for val in x])
