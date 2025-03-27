import numpy as np
from src.model import LambdaMART
from src.utils import create_example_data


def main():
    """
    This file provides a complete example of how to:
    1. Generate synthetic data for a learning-to-rank task.
    2. Split the data into training and testing sets.
    3. Initialize a LambdaMART model.
    4. Train the model on the training data.
    5. Make predictions on the test data.
    6. Evaluate the model's performance using NDCG.
    7. Visualize some example rankings.
    """

    # Generate example data.
    # n_samples is the total number of samples (n_queries * n_samples_per_query)
    # x is the feature matrix of shape (n_samples, n_features).
    # y is the relevance score of shape (n_samples,).
    # query_ids is the query identifier for each sample of shape (n_samples,).
    x, y, query_ids = create_example_data(
        n_queries=1,  # Number of queries
        n_features=2,  # Number of features
        n_samples_per_query=100  # Number of samples per query
    )

    # Split data into training 80% and testing 20% sets
    train_idx = np.random.rand(len(x)) < 0.8
    x_train, x_test = x[train_idx], x[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]
    query_ids_train, query_ids_test = query_ids[train_idx], query_ids[~train_idx]

    # Initialize and train LambdaMART
    model = LambdaMART(
        n_estimators=2,      # Number of trees
        learning_rate=0.05,     # Learning rate
        max_depth=5,           # Maximum depth of trees
        min_samples_split=2,   # Minimum samples per split
        random_state=32       # Random seed for reproducibility
    )

    print("Training LambdaMART model...")
    model.fit(x_train, y_train, query_ids_train)

    # Make predictions
    predictions = model.predict(x_test)
    print("Predictions:", predictions)
    print("True relevance:", y_test)

    # Evaluate the model
    ndcg_score = model.score(x_test, y_test, query_ids_test)
    print(f"NDCG@10 score: {ndcg_score:.4f}")

    # Print some example rankings
    print("\nExample rankings:")
    unique_queries = np.unique(query_ids_test)
    for query_id in unique_queries[:3]:  # Show rankings for first 3 queries
        query_mask = (query_ids_test == query_id)
        query_true = y_test[query_mask]
        query_pred = predictions[query_mask]

        # Sort by predicted scores
        sorted_idx = np.argsort(query_pred)[::-1]
        sorted_true = query_true[sorted_idx]

        print(f"\nQuery {query_id}:")
        print("True relevance | Predicted score")
        for i in range(min(7, len(sorted_true))):  # Show top 7 items
            print(f"{sorted_true[i]:<15} {query_pred[sorted_idx[i]]:.4f}")


if __name__ == "__main__":
    main()
