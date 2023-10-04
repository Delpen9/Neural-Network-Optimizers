# Data Science Libraries
import numpy as np
import pandas as pd
import mlrose_hiive as mlrose

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt


def optimization_algorithm_fitness_per_iteration(
    algorithm: str = "rhc", problem_type: str = "FourPeaks", iterations: int = 50
) -> tuple[np.ndarray, float, np.ndarray]:
    assert algorithm in [
        "rhc",
        "sa",
        "ga",
        "mimic",
    ], "Algorithm must be in ['rhc', 'sa', 'ga', 'mimic']"
    assert problem_type in [
        "FourPeaks",
        "OneMax",
        "FlipFlop",
    ], "Problem type must be in ['FourPeaks', 'OneMax', 'FlipFlop']"

    max_attempts_to_ensure_iterations = iterations * 10

    if problem_type == "FourPeaks":
        fitness = mlrose.FourPeaks(t_pct=0.1)
    elif problem_type == "OneMax":
        fitness = mlrose.OneMax()
    elif problem_type == "FlipFlop":
        fitness = mlrose.FlipFlop()

    problem = mlrose.DiscreteOpt(
        length=100, fitness_fn=fitness, maximize=True, max_val=2
    )

    if algorithm == "rhc":
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(
            problem,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            restarts=0,
            init_state=None,
            curve=True,
        )
    elif algorithm == "sa":
        schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.95, min_temp=0.001)
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
            problem,
            schedule=schedule,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            init_state=None,
            curve=True,
        )
    elif algorithm == "ga":
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
            problem,
            pop_size=200,
            mutation_prob=0.1,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            curve=True,
        )
    elif algorithm == "mimic":
        best_state, best_fitness, fitness_curve = mlrose.mimic(
            problem,
            pop_size=20,
            keep_pct=0.2,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            curve=True,
        )

    return (best_state, best_fitness, fitness_curve)


def get_optimization_algorithm_fitness_per_iteration_comparison(
    problem_type: str = "FourPeaks",
    iterations: int = 20,
    output_location: str = "../outputs/optimization_algorithms/",
    verbose: bool = True,
) -> None:
    assert problem_type in [
        "FourPeaks",
        "OneMax",
        "FlipFlop",
    ], "Problem type must be in ['FourPeaks', 'OneMax', 'FlipFlop']"

    algorithms = [
        "rhc",
        "sa",
        "ga",
        "mimic",
    ]
    fitness_per_iteration_list = []
    for algorithm in algorithms:
        _, best_fitness, curve = optimization_algorithm_fitness_per_iteration(
            algorithm, problem_type, iterations
        )
        if verbose == True:
            print(rf"Finished running algorithm: {algorithm}.")

        fitness_history = curve[:, 0].copy().reshape(-1, 1)
        fitness_per_iteration_list.append(fitness_history)

    fitness_per_iteration_np = np.hstack(tuple(fitness_per_iteration_list))
    fitness_per_iteration_df = pd.DataFrame(
        fitness_per_iteration_np, columns=algorithms
    )

    fitness_per_iteration_df.to_csv(
        rf"{output_location}/all_algorithms_fitness_per_iteration_{problem_type}",
        index=False,
    )


def get_optimization_algorithm_fitness_per_iteration_graphs(
    problem_type: str = "FourPeaks",
    input_location: str = "../outputs/optimization_algorithms/",
    output_location: str = "../outputs/optimization_algorithms/",
) -> None:
    df = pd.read_csv(
        rf"{input_location}/all_algorithms_fitness_per_iteration_{problem_type}"
    )

    df["iteration"] = df.index

    df_melted = df.melt(
        id_vars=["iteration"], var_name="algorithm", value_name="fitness"
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x="iteration", y="fitness", hue="algorithm")
    plt.title(rf"{problem_type}: Performance per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend(title="Algorithm")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(
        rf"{output_location}/all_algorithms_fitness_per_iteration_graph_{problem_type}.png"
    )


def get_all_optimization_algorithm_fitness_per_iteration_graphs() -> None:
    problem_types = ["FourPeaks", "OneMax", "FlipFlop"]

    for problem_type in problem_types:
        get_optimization_algorithm_fitness_per_iteration_comparison(problem_type)

    for problem_type in problem_types:
        get_optimization_algorithm_fitness_per_iteration_graphs(problem_type)


def optimization_algorithm_fitness_adjustable_problem_size(
    algorithm: str = "rhc",
    problem_type: str = "FourPeaks",
    iterations: int = 50,
    problem_size: int = 100,
) -> float:
    assert algorithm in [
        "rhc",
        "sa",
        "ga",
        "mimic",
    ], "Algorithm must be in ['rhc', 'sa', 'ga', 'mimic']"
    assert problem_type in [
        "FourPeaks",
        "OneMax",
        "FlipFlop",
    ], "Problem type must be in ['FourPeaks', 'OneMax', 'FlipFlop']"

    if problem_type == "FourPeaks":
        fitness = mlrose.FourPeaks(t_pct=0.1)
    elif problem_type == "OneMax":
        fitness = mlrose.OneMax()
    elif problem_type == "FlipFlop":
        fitness = mlrose.FlipFlop()

    max_attempts_to_ensure_iterations = iterations * 10

    problem = mlrose.DiscreteOpt(
        length=problem_size, fitness_fn=fitness, maximize=True, max_val=2
    )

    if algorithm == "rhc":
        _, best_fitness, _ = mlrose.random_hill_climb(
            problem,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            restarts=0,
            init_state=None,
            curve=True,
        )
    elif algorithm == "sa":
        schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.95, min_temp=0.001)
        _, best_fitness, fitness_curve = mlrose.simulated_annealing(
            problem,
            schedule=schedule,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            init_state=None,
            curve=True,
        )
    elif algorithm == "ga":
        _, best_fitness, _ = mlrose.genetic_alg(
            problem,
            pop_size=200,
            mutation_prob=0.1,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            curve=True,
        )
    elif algorithm == "mimic":
        _, best_fitness, _ = mlrose.mimic(
            problem,
            pop_size=20,
            keep_pct=0.2,
            max_attempts=max_attempts_to_ensure_iterations,
            max_iters=iterations,
            curve=True,
        )

    return best_fitness


def optimization_algorithm_fitness_per_problem_size_comparison(
    problem_type: str = "FourPeaks",
    iterations: int = 50,
) -> tuple[np.ndarray, float, np.ndarray]:
    assert problem_type in [
        "FourPeaks",
        "OneMax",
        "FlipFlop",
    ], "Problem type must be in ['FourPeaks', 'OneMax', 'FlipFlop']"

    algorithms = ["rhc", "sa", "ga", "mimic"]
    problem_sizes = np.arange(10, 110, 10).astype(int)

    fitness_per_problem_size = []
    fitness_per_problem_size_per_algorithm = []
    for algorithm in algorithms:
        for problem_size in problem_sizes:
            best_fitness = optimization_algorithm_fitness_adjustable_problem_size(
                algorithm,
                problem_type,
                iterations,
                problem_size,
            )
            fitness_per_problem_size.append(best_fitness)
        fitness_per_problem_size_per_algorithm.append(
            np.array(fitness_per_problem_size).reshape(-1, 1)
        )

    fitness_per_problem_size_per_algorithm_np = np.hstack(
        tuple(fitness_per_problem_size_per_algorithm)
    )
    fitness_per_problem_size_per_algorithm_df = pd.DataFrame(
        fitness_per_problem_size_per_algorithm_np, columns=algorithms
    )

    fitness_per_iteration_df.to_csv(
        rf"{output_location}/all_algorithms_fitness_per_problem_size_{problem_type}",
        index=False,
    )


def get_optimization_algorithm_fitness_per_problem_size_graphs(
    problem_type: str = "FourPeaks",
    input_location: str = "../outputs/optimization_algorithms/",
    output_location: str = "../outputs/optimization_algorithms/",
) -> None:
    df = pd.read_csv(
        rf"{input_location}/all_algorithms_fitness_per_problem_size_{problem_type}"
    )

    df["iteration"] = df.index

    df_melted = df.melt(
        id_vars=["iteration"], var_name="algorithm", value_name="fitness"
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x="iteration", y="fitness", hue="algorithm")
    plt.title(rf"{problem_type}: Performance per Problem Size")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend(title="Algorithm")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(
        rf"{output_location}/all_algorithms_fitness_per_problem_size_graph_{problem_type}.png"
    )


def get_all_optimization_algorithm_fitness_per_problem_size_graphs() -> None:
    problem_types = ["FourPeaks", "OneMax", "FlipFlop"]

    for problem_type in problem_types:
        optimization_algorithm_fitness_per_problem_size_comparison(problem_type)

    for problem_type in problem_types:
        get_optimization_algorithm_fitness_per_problem_size_graphs(problem_type)


if __name__ == "__main__":
    # get_all_optimization_algorithm_fitness_per_iteration_graphs()
    get_all_optimization_algorithm_fitness_per_problem_size_graphs()
