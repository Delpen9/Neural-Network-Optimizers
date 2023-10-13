# Data Science Libraries
import numpy as np
import pandas as pd
import mlrose_hiive as mlrose

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn Libraries
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

# Python Standard Libraries
import time

from dataset_preprocessing import (
    preprocess_datasets,
)


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    learning_rate: float = 0.01,
    hidden_nodes: int = 10,
    num_training_iterations: int = 1000,
    algorithm: str = "gradient_descent",
    activation: str = "relu",
    random_state: int = 42,
    **kwargs,
) -> tuple[list[float], list[float]]:
    assert algorithm in [
        "gradient_descent",
        "simulated_annealing",
        "random_hill_climb",
        "genetic_alg",
        "mimic",
    ], "algorithm parameter must be a value in ['gradient_descent', 'simulated_annealing', 'random_hill_climb', 'genetic_alg', 'mimic']"

    assert (
        num_training_iterations % 10 == 0
    ), "num_training_iterations should be some multiple of 10"

    model = mlrose.NeuralNetwork(
        hidden_nodes=[hidden_nodes],
        activation="relu",
        algorithm=algorithm,
        max_iters=num_training_iterations,
        bias=True,
        is_classifier=True,
        learning_rate=learning_rate,
        early_stopping=True,
        clip_max=5,
        max_attempts=num_training_iterations,
        random_state=random_state,
        **kwargs,
    )

    training_losses = []
    validation_losses = []

    validation_iteration_samples = np.arange(
        np.floor(num_training_iterations // 10),
        num_training_iterations + np.ceil(num_training_iterations // 10),
        np.floor(num_training_iterations // 10),
    ).astype(int)

    for validation_iteration_sample in validation_iteration_samples:
        model.max_iters = validation_iteration_sample
        model.fit(X_train, y_train)

        # Training Loss
        model.predict(X_train)
        y_train_pred_prob = model.predicted_probs
        training_loss = log_loss(y_train, y_train_pred_prob)
        training_losses.append(training_loss)

        # Validation Loss
        model.predict(X_val)
        y_val_pred_prob = model.predicted_probs
        validation_loss = log_loss(y_val, y_val_pred_prob)
        validation_losses.append(validation_loss)

        print(
            rf"Iteration: {validation_iteration_sample}; Training Loss: {training_loss}; Validation Loss: {validation_loss}"
        )

    return (training_losses, validation_losses)


if __name__ == "__main__":
    (
        # Auction
        auction_train_X,
        auction_train_y,
        auction_val_X,
        auction_val_y,
        auction_test_X,
        auction_test_y,
        # Dropout
        dropout_train_X,
        dropout_train_y,
        dropout_val_X,
        dropout_val_y,
        dropout_test_X,
        dropout_test_y,
    ) = preprocess_datasets()

    algorithm = "genetic_alg"
    kwargs = {}
    if algorithm == "genetic_alg":
        kwargs = {"pop_size": 10, "mutation_prob": 0.1}
    elif algorithm == "simulated_annealing":
        kwargs = {"schedule": mlrose.GeomDecay()}
    elif algorithm == "random_hill_climb":
        kwargs = {"restarts": 50}

    
    (training_losses, validation_losses) = train_neural_network(
        auction_train_X,
        auction_train_y.iloc[:, 0],
        auction_val_X,
        auction_val_y.iloc[:, 0],
        learning_rate=1e-1,
        hidden_nodes=2,
        num_training_iterations=2500,
        algorithm=algorithm,
        activation="relu",
        random_state=42,
        **kwargs
    )
