"""
src/training/model.py

Models:
    - MLP
    - LSTM
    - Spark ML Random Forest
"""

import torch.nn as nn

from pyspark.ml.classification import RandomForestClassifier


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# LSTM
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Linear(
            32,
            2
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        return self.fc(
            hidden[-1]
        )


# ============================================================
# Random Forest
# ============================================================

def get_random_forest(
    num_trees,
    max_depth
):

    return RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42
    )