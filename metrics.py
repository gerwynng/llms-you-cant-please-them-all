import itertools
import random

import numpy as np
from tqdm import tqdm


def compute_metric(grades):
    grades = np.array(grades)
    avg_h = np.mean(np.std(grades, axis=1))
    avg_q = np.mean(np.mean(grades, axis=1))
    min_v = min(np.std(grades, axis=0))
    final_metric = (avg_h * min_v) / (9 - avg_q)
    return avg_h, avg_q, min_v, final_metric


def exact_search(target_metric, num_rows=2, num_cols=3, fixed_rows=None):
    # Allowed grades
    allowed_grades = [0, 1, 8, 9]

    # Generate all possible combinations of grades for free rows
    num_free_rows = num_rows - (len(fixed_rows) if fixed_rows else 0)
    all_combinations = itertools.product(
        allowed_grades, repeat=num_free_rows * num_cols
    )

    best_grades = None
    best_loss = float("inf")

    for combination in tqdm(all_combinations):
        # Build the current grades matrix
        current_grades = []
        if fixed_rows:
            current_grades.extend(fixed_rows)
        current_grades.extend(
            np.array(combination).reshape(
                (
                    num_free_rows,
                    num_cols,
                )
            )
        )
        current_grades = np.array(current_grades)

        # Compute the metric for the current grades
        _, _, _, current_metric = compute_metric(current_grades)
        current_loss = abs(current_metric - target_metric)

        # Update the best solution
        if current_loss < best_loss:
            best_grades = current_grades
            best_loss = current_loss

        # Early stopping if loss is very low
        if best_loss < 1e-6:
            break

    return best_grades, best_loss


def simulated_annealing(
    target_metric,
    num_rows=2,
    num_cols=3,
    max_iter=10000,
    temp=1.0,
    cooling_rate=0.99,
    fixed_rows=None,
):
    # Allowed grades
    allowed_grades = [0, 9]

    # Initialize grades
    fixed_rows = fixed_rows or []
    num_free_rows = num_rows - len(fixed_rows)

    # Randomly initialize the free part of the grades matrix
    free_grades = np.random.choice(
        allowed_grades,
        size=(
            num_free_rows,
            num_cols,
        ),
    )
    current_grades = np.vstack(fixed_rows + [free_grades])
    _, _, _, current_metric = compute_metric(current_grades)
    current_loss = abs(current_metric - target_metric)

    best_grades = current_grades.copy()
    best_loss = current_loss

    for i in range(max_iter):
        # Generate a neighbor by tweaking one random grade in the free rows
        free_grades = current_grades[len(fixed_rows) :].copy()
        row, col = random.randint(0, num_free_rows - 1), random.randint(
            0,
            num_cols - 1,
        )
        free_grades[row, col] = random.choice(allowed_grades)

        # Rebuild the grades matrix
        new_grades = np.vstack(fixed_rows + [free_grades])

        # Compute the metric for the new grades
        _, _, _, new_metric = compute_metric(new_grades)
        new_loss = abs(new_metric - target_metric)

        # Decide whether to accept the new solution
        if new_loss < current_loss or random.random() < np.exp(
            -(new_loss - current_loss) / temp
        ):
            current_grades = new_grades
            current_loss = new_loss

            # Update the best solution
            if new_loss < best_loss:
                best_grades = new_grades
                best_loss = new_loss

        # Cool down
        temp *= cooling_rate

        # Early stopping if loss is very low
        if best_loss < 1e-6:
            break

    return best_grades, best_loss


if __name__ == "__main__":
    target_metric = 4.139  # 3.882  # 3.440
    num_rows = 3
    num_cols = 3
    max_iter = 500000

    # best_grades, best_loss = simulated_annealing(
    #     target_metric,
    #     num_rows=num_rows,
    #     num_cols=num_cols,
    #     max_iter=max_iter,
    # )
    best_grades, best_loss = exact_search(
        target_metric,
        num_rows=num_rows,
        num_cols=num_cols,
        fixed_rows=[[0, 9, 9], [9, 0, 0]],
    )
    print("Best Grades Found:")
    print(best_grades)
    # print("Loss:", best_loss)

    # Print metrics for the best grades
    avg_h, avg_q, min_v, final_metric = compute_metric(best_grades)
    print("\nMetrics for the Best Grades:")
    print(f"avg_h (Average Horizontal Std): {avg_h:.3f}")
    print(f"avg_q (Average Quality): {avg_q:.3f}")
    print(f"min_v (Minimum Vertical Std): {min_v:.3f}")
    print(f"Final Metric: {final_metric:3f}")
