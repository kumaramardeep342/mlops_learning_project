"""
Models for Predictive Maintenance
"""

import torch
import torch.nn as nn
from pyspark.ml.classification import RandomForestClassifier


# ============================================================
# MLP
# ============================================================


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# LSTM
# ============================================================


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])


# ============================================================
# Random Forest
# ============================================================


def get_random_forest(num_trees=50, max_depth=5):
    return RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42,
    )
