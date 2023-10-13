# Data Science Libraries
import numpy as np
import pandas as pd

from dataset_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)

from optimization_problems import (
    get_all_optimization_algorithm_fitness_per_iteration_graphs,
    get_all_optimization_algorithm_fitness_per_problem_size_graphs,
    get_all_optimization_algorithm_fitness_per_evaluation_graphs,
    get_all_optimization_algorithm_fitness_per_wall_clock_time_graphs,
    get_performance_difference_between_random_hill_climbing_and_simulated_annealing,
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


    get_all_optimization_algorithm_fitness_per_iteration_graphs()
    get_all_optimization_algorithm_fitness_per_problem_size_graphs(iterations = 5)
    get_all_optimization_algorithm_fitness_per_evaluation_graphs()
    get_all_optimization_algorithm_fitness_per_wall_clock_time_graphs()
    get_performance_difference_between_random_hill_climbing_and_simulated_annealing()