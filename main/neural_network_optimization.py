# Data Science Libraries
import numpy as np
import pandas as pd
import mlrose_hiive as mlrose

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn Libraries
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

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
    validation_iteration_samples: int = None,
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
        clip_max=1,
        max_attempts=num_training_iterations,
        random_state=random_state,
        **kwargs,
    )

    training_losses = []
    validation_losses = []

    if validation_iteration_samples == None:
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


def get_neural_network_optimization_performance_table(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
    algorithm: str = "genetic_alg",
    validation_iteration_samples: list = [500, 2500, 5000, 10000],
    filename: str = "",
    output_location: str = "../outputs/neural_network_optimization/",
) -> None:
    assert filename != "", "filename parameter must not be empty."

    kwargs = {}
    if algorithm == "genetic_alg":
        kwargs = {"pop_size": 10, "mutation_prob": 0.1}
    elif algorithm == "simulated_annealing":
        kwargs = {"schedule": mlrose.GeomDecay()}
    elif algorithm == "random_hill_climb":
        kwargs = {"restarts": 50}

    (training_losses, validation_losses) = train_neural_network(
        auction_train_X,
        auction_train_y,
        auction_val_X,
        auction_val_y,
        learning_rate=1e-3,
        hidden_nodes=2,
        num_training_iterations=2500,
        algorithm=algorithm,
        activation="relu",
        random_state=42,
        validation_iteration_samples=validation_iteration_samples,
        **kwargs,
    )

    validation_iteration_samples = np.array([validation_iteration_samples]).T
    training_losses = np.array([training_losses]).T
    validation_losses = np.array([validation_losses]).T

    nn_performance_history = pd.DataFrame(
        np.hstack((validation_iteration_samples, training_losses, validation_losses)),
        columns=["Iterations", "Training Loss", "Validation Loss"],
    )

    output_path = rf"{output_location}{filename}"
    nn_performance_history.to_csv(output_path)


def get_all_performance_tables(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
) -> None:
    get_neural_network_optimization_performance_table(
        train_X,
        train_y,
        val_X,
        val_y,
        algorithm="random_hill_climb",
        validation_iteration_samples=[100, 200, 500, 1000, 1500],
        filename="random_hill_climb_performance_per_iteration.csv",
        output_location="../outputs/neural_network_optimization/",
    )

    get_neural_network_optimization_performance_table(
        train_X,
        train_y,
        val_X,
        val_y,
        algorithm="genetic_alg",
        validation_iteration_samples=[100, 200, 500, 1000, 1500],
        filename="genetic_alg_performance_per_iteration.csv",
        output_location="../outputs/neural_network_optimization/",
    )

    get_neural_network_optimization_performance_table(
        train_X,
        train_y,
        val_X,
        val_y,
        algorithm="simulated_annealing",
        validation_iteration_samples=[100, 200, 500, 1000, 1500],
        filename="simulated_annealing_performance_per_iteration.csv",
        output_location="../outputs/neural_network_optimization/",
    )

    get_neural_network_optimization_performance_table(
        train_X,
        train_y,
        val_X,
        val_y,
        algorithm="gradient_descent",
        validation_iteration_samples=[100, 200, 500, 1000, 1500],
        filename="gradient_descent_performance_per_iteration.csv",
        output_location="../outputs/neural_network_optimization/",
    )


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

    auction_train_y = auction_train_y.iloc[:, 0].to_numpy()
    auction_val_y = auction_val_y.iloc[:, 0].to_numpy()

    get_all_performance_tables(
        auction_train_X.values,
        auction_train_y,
        auction_val_X.values,
        auction_val_y,
    )
