import logging
import numpy as np
import pytest
from lambdamart.model import LambdaMART, evaluate_ndcg
from lambdamart.tree import DecisionTree
from lambdamart.utils import create_example_data, compute_lambdas

logging.basicConfig(level=logging.INFO, format='%(message)s')


@pytest.fixture(name="example_data", scope="session")
def fixture_example_data():
    """Generate example data for testing."""
    x, y, query_ids = create_example_data(
        n_queries=5,
        n_features=3,
        n_samples_per_query=7
    )
    return x, y, query_ids


def test_lambda_mart_initialization():
    """Test LambdaMART initialization."""
    model = LambdaMART(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )

    assert model.n_estimators == 10
    assert model.learning_rate == 0.1
    assert model.max_depth == 3
    assert model.min_samples_split == 2
    assert model.random_state == 42
    assert not model.trees


def test_lambda_mart_fit_predict(example_data):
    """Test LambdaMART fit and predict functionality."""
    x, y, query_ids = example_data
    model = LambdaMART(
        n_estimators=3,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2,
        random_state=42
    )

    # Test fit
    model.fit(x, y, query_ids)
    assert len(model.trees) == 3

    # Test predict
    predictions = model.predict(x)
    assert predictions.shape == (x.shape[0],)
    assert np.all(np.isfinite(predictions))


def test_lambda_mart_score(example_data):
    """Test LambdaMART scoring functionality."""
    x, y, query_ids = example_data
    model = LambdaMART(
        n_estimators=3,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2,
        random_state=42
    )

    model.fit(x, y, query_ids)
    score = model.score(x, y, query_ids)
    logging.info("Score: %f", score)
    #print(f"Printed_Score: {score}")
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_lambda_mart_edge_cases():
    """Test LambdaMART with edge cases."""
    # Test with single feature
    x = np.random.randn(100, 1)
    y = np.random.randint(0, 5, size=100)
    query_ids = np.repeat(np.arange(10), 10)

    model = LambdaMART(
        n_estimators=3,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2,
        random_state=42
    )

    model.fit(x, y, query_ids)
    predictions = model.predict(x)
    assert predictions.shape == (100,)

    # Test with single query
    x = np.random.randn(10, 5)
    y = np.random.randint(0, 5, size=10)
    query_ids = np.zeros(10)

    model.fit(x, y, query_ids)
    predictions = model.predict(x)
    assert predictions.shape == (10,)


def test_ndcg():
    """Test NDCG calculation."""
    true_relevance = np.array([3, 2, 3, 0, 1, 2])
    predicted_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    query_ids = np.array([0, 0, 0, 1, 1, 1])

    ndcg_score = evaluate_ndcg(true_relevance, predicted_scores, query_ids)
    assert isinstance(ndcg_score, float)
    assert 0 <= ndcg_score <= 1


def test_lambdas():
    """Test lambda computation."""
    scores = np.array([0.9, 0.8, 0.7])
    true_relevance = np.array([3, 2, 1])
    query_ids = np.array([0, 0, 0])

    lambdas, weights = compute_lambdas(scores, true_relevance, query_ids)

    assert lambdas.shape == (3,)
    assert weights.shape == (3,)
    assert np.all(np.isfinite(lambdas))
    assert np.all(np.isfinite(weights))


def test_decision_tree():
    """Test DecisionTree functionality."""
    x = np.random.randn(100, 5)
    y = np.random.randn(100)
    weights = np.ones(100)

    tree = DecisionTree(
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )

    tree.fit(x, y, weights)
    predictions = tree.predict(x)

    assert predictions.shape == (100,)
    assert np.all(np.isfinite(predictions))


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    x, y, query_ids = create_example_data(
        n_queries=10,
        n_features=5,
        n_samples_per_query=10
    )

    model = LambdaMART()

    # Test mismatched input sizes
    with pytest.raises(ValueError):
        model.fit(x[:-1], y, query_ids)

    with pytest.raises(ValueError):
        model.fit(x, y[:-1], query_ids)

    with pytest.raises(ValueError):
        model.fit(x, y, query_ids[:-1])

    # Test invalid parameter values
    with pytest.raises(ValueError):
        LambdaMART(n_estimators=0)

    with pytest.raises(ValueError):
        LambdaMART(learning_rate=-0.1)

    with pytest.raises(ValueError):
        LambdaMART(max_depth=-1)

    with pytest.raises(ValueError):
        LambdaMART(min_samples_split=1)
