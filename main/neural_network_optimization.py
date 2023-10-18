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
from sklearn.metrics import accuracy_score, roc_auc_score

# Python Standard Libraries
import time

from dataset_preprocessing import (
    preprocess_datasets,
)


def alternative_train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    experiment_algorithm: str = "simulated_annealing",
) -> tuple[any, any]:
    grid_search_parameters = {
        "max_iters": [1, 2, 4, 8, 16, 32, 64, 128
        ],
        "learning_rate": [1e+1, 1e-1, 1e-2, 1e-3],
        "schedule": [
            mlrose.ArithDecay(1),
            mlrose.ArithDecay(100),
            mlrose.ArithDecay(1000),
        ],
        "activation": [mlrose.neural.activation.relu],
    }

    if experiment_algorithm == "experiment_algorithm":
        algorithm = mlrose.algorithms.sa.simulated_annealing
        iteration_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    elif experiment_algorithm == "random_hill_climb":
        algorithm = mlrose.algorithms.rhc.random_hill_climb
        iteration_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    elif experiment_algorithm == "genetic_algorithm":
        algorithm = mlrose.algorithms.ga.genetic_alg
        algorithm.pop_size = 10
        iteration_list = np.arange(1, 11, 1).astype(int).tolist()
    elif experiment_algorithm == "gradient_descent":
        algorithm = mlrose.algorithms.gd.gradient_descent
        iteration_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    model = mlrose.NNGSRunner(
        x_train=X_train,
        y_train=y_train,
        x_test=X_val,
        y_test=y_val,
        experiment_name=experiment_algorithm,
        algorithm=algorithm,
        grid_search_parameters=grid_search_parameters,
        iteration_list=iteration_list,
        hidden_layer_sizes=[[50, 50]],
        bias=True,
        early_stopping=True,
        clip_max=1e+3,
        max_attempts=500,
        generate_curves=True,
        seed=42,
        n_jobs=-1,
    )

    (fitness_results, fitness_curves) = model.run()[0:2]

    return (fitness_results, fitness_curves)


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
    training_accuracies = []
    training_aucs = []

    validation_losses = []
    validation_accuracies = []
    validation_aucs = []

    if validation_iteration_samples == None:
        validation_iteration_samples = np.arange(
            np.floor(num_training_iterations // 10),
            num_training_iterations + np.ceil(num_training_iterations // 10),
            np.floor(num_training_iterations // 10),
        ).astype(int)

    for validation_iteration_sample in validation_iteration_samples:
        model.max_iters = validation_iteration_sample
        model.fit(X_train, y_train)

        # Training Metrics
        y_train_pred = model.predict(X_train)
        y_train_pred_prob = model.predicted_probs
        training_loss = log_loss(y_train, y_train_pred_prob)
        training_accuracy_score = accuracy_score(y_train, y_train_pred)
        training_auc_score = roc_auc_score(y_train, y_train_pred_prob)

        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy_score)
        training_aucs.append(training_auc_score)

        # Validation Metrics
        y_val_pred = model.predict(X_val)
        y_val_pred_prob = model.predicted_probs
        validation_loss = log_loss(y_val, y_val_pred_prob)
        validation_accuracy_score = accuracy_score(y_val, y_val_pred)
        validation_auc_score = roc_auc_score(y_val, y_val_pred_prob)

        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy_score)
        validation_aucs.append(validation_auc_score)

        print(
            rf"""Iteration: {validation_iteration_sample};
            Training Loss: {training_loss}; Validation Loss: {validation_loss};
            Training Accuracy: {training_accuracy_score}; Validation Accuracy: {validation_accuracy_score};
            Training AUC: {training_auc_score}; Validation AUC: {validation_auc_score}"""
        )

    return (
        training_losses,
        validation_losses,
        training_accuracies,
        validation_accuracies,
        training_aucs,
        validation_aucs,
    )


def get_neural_network_optimization_performance_table(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
    algorithm: str = "genetic_alg",
    validation_iteration_samples: list = [500, 2500, 5000, 10000],
    filename: str = "",
    output_location: str = "../outputs/neural_network_optimization/",
    dataset_type: str = "auction",
) -> None:
    assert filename != "", "filename parameter must not be empty."

    kwargs = {}
    if algorithm == "genetic_alg":
        kwargs = {"pop_size": 10, "mutation_prob": 0.1}
    elif algorithm == "simulated_annealing":
        kwargs = {"schedule": mlrose.GeomDecay()}
    elif algorithm == "random_hill_climb":
        kwargs = {"restarts": 50}

    (
        training_losses,
        validation_losses,
        training_accuracies,
        validation_accuracies,
        training_aucs,
        validation_aucs,
    ) = train_neural_network(
        auction_train_X,
        auction_train_y,
        auction_val_X,
        auction_val_y,
        learning_rate=1e-3,
        hidden_nodes=100,
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

    training_accuracies = np.array([training_accuracies]).T
    validation_accuracies = np.array([validation_accuracies]).T

    training_aucs = np.array([training_aucs]).T
    validation_aucs = np.array([validation_aucs]).T

    nn_performance_history = pd.DataFrame(
        np.hstack(
            (
                validation_iteration_samples,
                training_losses,
                validation_losses,
                training_accuracies,
                validation_accuracies,
                training_aucs,
                validation_aucs,
            )
        ),
        columns=[
            "Iterations",
            "Training Loss",
            "Validation Loss",
            "Training Accuracies",
            "Validation Accuracies",
            "Training AUCs",
            "Validation AUCs",
        ],
    )

    output_path = rf"{output_location}{dataset_type}_{filename}"
    nn_performance_history.to_csv(output_path)


def get_all_performance_graphs(
    output_location: str = "../outputs/neural_network_optimization/",
    dataset_type: str = "auction",
) -> None:
    algorithms = [
        "gradient_descent",
        "simulated_annealing",
        "random_hill_climb",
        "genetic_alg",
    ]
    neural_network_algorithm_peformance_tables = [
        rf"{output_location}{dataset_type}_{algorithm}_performance_per_iteration.csv"
        for algorithm in algorithms
    ]

    metrics = [
        ("Training Loss", "Validation Loss"),
        ("Training Accuracies", "Validation Accuracies"),
        ("Training AUCs", "Validation AUCs"),
    ]
    colors = ["blue", "green", "red"]

    for performance_table_path, algorithm in zip(
        neural_network_algorithm_peformance_tables, algorithms
    ):
        performance_df = pd.read_csv(performance_table_path, index_col=0)

        plt.figure(figsize=(12, 8))

        for (train_metric, val_metric), color in zip(metrics, colors):
            plt.plot(
                performance_df["Iterations"],
                performance_df[train_metric],
                label=train_metric,
                color=color,
            )
            plt.plot(
                performance_df["Iterations"],
                performance_df[val_metric],
                label=val_metric,
                linestyle="--",
                color=color,
            )

        plt.xlabel("Iterations")
        plt.ylabel("Performance Metrics")
        plt.title(
            rf"{' '.join(word.capitalize() for word in algorithm.split('_'))}: Performance over Iterations"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            rf"{output_location}{dataset_type}_{algorithm}_performance_per_iteration_graph.png"
        )


def get_all_performance_tables(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
    dataset_type: str = "auction",
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
        dataset_type=dataset_type,
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
        dataset_type=dataset_type,
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
        dataset_type=dataset_type,
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
        dataset_type=dataset_type,
    )


def get_specific_performance_graphs(
    output_location: str = "../outputs/neural_network_optimization/",
    dataset_type: str = "auction",
) -> None:
    algorithms = ["gradient_descent", "genetic_alg"]
    neural_network_algorithm_peformance_tables = [
        rf"{output_location}{dataset_type}_{algorithm}_performance_per_iteration_low_iteration_count.csv"
        for algorithm in algorithms
    ]

    metrics = [
        ("Training Loss", "Validation Loss"),
        ("Training Accuracies", "Validation Accuracies"),
        ("Training AUCs", "Validation AUCs"),
    ]
    colors = ["blue", "green", "red"]

    for performance_table_path, algorithm in zip(
        neural_network_algorithm_peformance_tables, algorithms
    ):
        performance_df = pd.read_csv(performance_table_path, index_col=0)

        plt.figure(figsize=(12, 8))

        for (train_metric, val_metric), color in zip(metrics, colors):
            plt.plot(
                performance_df["Iterations"],
                performance_df[train_metric],
                label=train_metric,
                color=color,
            )
            plt.plot(
                performance_df["Iterations"],
                performance_df[val_metric],
                label=val_metric,
                linestyle="--",
                color=color,
            )

        plt.xlabel("Iterations")
        plt.ylabel("Performance Metrics")
        plt.title(
            rf"{' '.join(word.capitalize() for word in algorithm.split('_'))}: Performance over Iterations"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            rf"{output_location}{dataset_type}_{algorithm}_performance_per_iteration_graph_low_iteration_count.png"
        )


def get_specific_performance_tables(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
    dataset_type: str = "auction",
    validation_iteration_samples: list = [20, 40, 60, 80, 100],
) -> None:
    get_neural_network_optimization_performance_table(
        train_X,
        train_y,
        val_X,
        val_y,
        algorithm="genetic_alg",
        validation_iteration_samples=validation_iteration_samples,
        filename="genetic_alg_performance_per_iteration_low_iteration_count.csv",
        output_location="../outputs/neural_network_optimization/",
        dataset_type=dataset_type,
    )

    get_neural_network_optimization_performance_table(
        train_X,
        train_y,
        val_X,
        val_y,
        algorithm="gradient_descent",
        validation_iteration_samples=validation_iteration_samples,
        filename="gradient_descent_performance_per_iteration_low_iteration_count.csv",
        output_location="../outputs/neural_network_optimization/",
        dataset_type=dataset_type,
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
    auction_test_y = auction_test_y.iloc[:, 0].to_numpy()

    dropout_train_y = dropout_train_y.to_numpy()
    dropout_val_y = dropout_val_y.to_numpy()
    dropout_test_y = dropout_test_y.to_numpy()

    # get_all_performance_tables(
    #     auction_train_X.values,
    #     auction_train_y,
    #     auction_val_X.values,
    #     auction_val_y,
    #     dataset_type="auction",
    # )

    # get_all_performance_graphs(dataset_type = "auction")

    # get_specific_performance_tables(
    #     auction_train_X.values,
    #     auction_train_y,
    #     auction_val_X.values,
    #     auction_val_y,
    #     dataset_type="auction",
    # )

    # get_specific_performance_graphs(dataset_type = "auction")

    (fitness_results, fitness_curves) = alternative_train_neural_network(
        auction_train_X.values,
        auction_train_y,
        auction_val_X.values,
        auction_val_y,
        experiment_algorithm="simulated_annealing",
    )

    print(fitness_curves)