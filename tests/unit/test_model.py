import pytest
import torch
from pyspark.sql import SparkSession
from src.training.model import MLP, LSTMModel, get_random_forest

# ============================================================
# MLP Model Tests
# ============================================================

def test_mlp_initialization():
    """Test if MLP initializes without errors."""
    model = MLP(input_size=5)
    assert model is not None, "MLP model failed to initialize"


def test_mlp_output_shape():
    """Test if MLP outputs shape (batch_size, 2) for tabular features."""
    model = MLP(input_size=5)
    model.eval()

    dummy_input = torch.randn(4, 5)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2), f"Expected shape (4, 2), but got {output.shape}"


def test_mlp_numerical_stability():
    """Test that MLP output contains no NaNs or Infs."""
    model = MLP(input_size=5)
    model.eval()

    dummy_input = torch.randn(2, 5)
    with torch.no_grad():
        output = model(dummy_input)

    assert not torch.isnan(output).any(), "MLP output contains NaN values"
    assert not torch.isinf(output).any(), "MLP output contains Inf values"


# ============================================================
# LSTM Model Tests
# ============================================================

def test_lstm_initialization():
    """Test if LSTMModel initializes without errors."""
    model = LSTMModel(input_size=5)
    assert model is not None, "LSTM model failed to initialize"


def test_lstm_output_shape():
    """Test if LSTM outputs shape (batch_size, 2) for sequential input."""
    model = LSTMModel(input_size=5)
    model.eval()

    dummy_input = torch.randn(4, 5, 5)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2), f"Expected shape (4, 2), but got {output.shape}"


def test_lstm_numerical_stability():
    """Test that LSTM output contains no NaNs or Infs."""
    model = LSTMModel(input_size=5)
    model.eval()

    dummy_input = torch.randn(2, 5, 5)
    with torch.no_grad():
        output = model(dummy_input)

    assert not torch.isnan(output).any(), "LSTM output contains NaN values"
    assert not torch.isinf(output).any(), "LSTM output contains Inf values"


# ============================================================
# Random Forest Tests (PySpark)
# ============================================================

@pytest.fixture(scope="module")
def spark_session():
    """Fixture to manage SparkSession lifecycle during testing."""
    spark = (
        SparkSession.builder.appName("TestRandomForest")
        .master("local[1]")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_random_forest_initialization():
    """Test if PySpark RandomForestClassifier initializes with parameters."""
    rf = get_random_forest(num_trees=10, max_depth=3)
    assert rf.getNumTrees() == 10, "Random Forest tree count mismatch"
    assert rf.getMaxDepth() == 3, "Random Forest max depth mismatch"


def test_random_forest_fit_and_predict(spark_session):
    """Test if Random Forest trains and outputs predictions on Spark DataFrames."""
    from pyspark.ml.feature import VectorAssembler

    # Create dummy spark dataframe with 5 features and 1 label
    data = [
        (300.0, 310.0, 1500.0, 40.0, 10.0, 0.0),
        (305.0, 315.0, 1400.0, 50.0, 200.0, 1.0),
    ]
    columns = [
        "air_temp",
        "proc_temp",
        "speed",
        "torque",
        "tool_wear",
        "label",
    ]
    df = spark_session.createDataFrame(data, columns)

    assembler = VectorAssembler(
        inputCols=["air_temp", "proc_temp", "speed", "torque", "tool_wear"],
        outputCol="features",
    )
    df_transformed = assembler.transform(df)

    rf = get_random_forest(num_trees=5, max_depth=2)
    model = rf.fit(df_transformed)

    predictions = model.transform(df_transformed)
    assert "prediction" in predictions.columns, "Missing 'prediction' column in output"
    assert predictions.count() == 2, "Output row count mismatch"