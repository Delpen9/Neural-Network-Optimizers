import mlrose_hiive as mlrose
import numpy as np


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

    return best_state, best_fitness, fitness_curve


def get_optimization_algorithm_fitness_per_iteration_comparison(
    problem_type: str = "FourPeaks",
    iterations: int = 50,
    output_location = "../outputs/optimization_algorithms/",
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
        fitness_history = curve[:, 0].copy()
        fitness_per_iteration_list.append(fitness_history)
    fitness_per_iteration_np = np.hstack(tuple(fitness_per_iteration_list))
    fitness_per_iteration_df = pd.DataFrame(
        fitness_per_iteration_np, columns=algorithms
    )

    fitness_per_iteration_df.to_csv(fr"{output_location}/all_algorithms_fitness_per_iteration_{problem_type}")


if __name__ == "__main__":
    algorithms = ["rhc", "sa", "ga", "mimic"]
    problem_types = ["FourPeaks", "OneMax", "FlipFlop"]

    for problem_type in problem_types:
        get_optimization_algorithm_fitness_per_iteration_comparison(problem_type)
