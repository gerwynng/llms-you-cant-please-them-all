import argparse
import itertools
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from utils.metrics import score


def build_combined_grades(
    x1: int,
    x2: int,
    x3: int,
) -> NDArray[Any]:
    """
    Build the combined grades array of shape (n, 3),
    consisting of:
        [9, 0, 9] repeated x1 times
        [0, 9, 9] repeated x2 times
        [9, 9, 0] repeated x3 times
    """
    grades1 = np.tile([9, 0, 9], (x1, 1))
    grades2 = np.tile([0, 9, 9], (x2, 1))
    grades3 = np.tile([9, 9, 0], (x3, 1))
    return np.vstack((grades1, grades2, grades3))


def find_x123_for_score(
    n: int, target_score: float, tolerance: float = 1e-4
) -> list[tuple[int, int, int, float]]:
    """
    Brute force search over x1, x2, x3 such that x1 + x2 + x3 = n.
    Return a list of (x1, x2, x3, achieved_score) for all combos whose
    score is within `tolerance` of `target_score`.
    """
    solutions: list[tuple[int, int, int, float]] = []
    for x1 in range(n + 1):
        for x2 in range(n + 1 - x1):
            x3 = n - x1 - x2
            # Build the array of grades
            combined = build_combined_grades(x1, x2, x3)
            # Compute the score
            sc = score(combined)
            # Check how close we are to the desired target_score
            if abs(sc - target_score) <= tolerance:
                solutions.append((x1, x2, x3, sc))
    return solutions


def get_single_swaps(x1: int, x2: int, x3: int):
    """
    Generator that yields all possible single swaps (i->j, k) from (x1,x2,x3).
    i->j means move k from x_i to x_j, for k = 1..x_i, i != j.
    We'll yield (new_x1, new_x2, new_x3, i->j, k).
    """
    original = [x1, x2, x3]
    for i, j in itertools.permutations([0, 1, 2], 2):  # i!=j
        if original[i] == 0:
            continue
        for k in range(1, original[i] + 1):
            newvals = original[:]
            newvals[i] -= k
            newvals[j] += k
            verbose = f"{i + 1}->{j + 1}"
            yield (newvals[0], newvals[1], newvals[2], verbose, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find (x1, x2, x3) for a given target score."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Total number of samples (n = x1 + x2 + x3)",
    )
    parser.add_argument(
        "--target_score",
        type=float,
        help="Target score to find combinations for",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for matching the target score (default: 1e-4)",
    )

    args = parser.parse_args()

    combos = find_x123_for_score(
        args.n_samples,
        args.target_score,
        tolerance=args.tolerance,
    )
    if combos:
        print(f"total {len(combos)} combinations found!")
        for xx1, xx2, xx3, sc in combos:
            print(f"x1={xx1}, x2={xx2}, x3={xx3}, Score={sc}")
    else:
        print("No combinations found within given tolerance.")

    # 1) Collect baseline scores for each of the combos
    combos = [(x1, x2, x3) for x1, x2, x3, _ in combos]

    baselines: list[float] = []
    for x1, x2, x3 in combos:
        arr = build_combined_grades(x1, x2, x3)
        sc = score(arr)
        baselines.append(sc)

    improvements: dict[Any, Any] = defaultdict(lambda: [0.0] * len(combos))

    # (best_improvement, best i->j, best k)
    best_per_combo: list[Any] = []

    baseline_sc = -1

    for idx, (x1, x2, x3) in enumerate(combos):
        baseline_sc: float = baselines[idx]
        best_impr = float("-inf")
        best_swap = None

        for nx1, nx2, nx3, move_str, k in get_single_swaps(x1, x2, x3):
            new_arr = build_combined_grades(nx1, nx2, nx3)
            new_sc = score(new_arr)
            impr = new_sc - baseline_sc

            # Add to improvements dictionary
            improvements[(move_str, k)][idx] = impr

            # Track best single swap for this particular baseline
            if impr > best_impr:
                best_impr = impr
                best_swap = (move_str, k)

        best_per_combo.append((best_impr, best_swap))

    # We have improvements dict:  key=(move_str, k)
    results: list[tuple[Any, Any, Any]] = []
    for move_key, impr_list in improvements.items():
        impr_arr = np.array(impr_list)
        avg_impr = np.mean(impr_arr)
        min_impr = np.min(impr_arr)
        results.append((move_key, avg_impr, min_impr))

    # Sort by best average improvement:
    results_by_avg = sorted(results, key=lambda x: x[1], reverse=True)
    best_by_avg = results_by_avg[0]

    # Sort by best worst-case (i.e. best min improvement):
    results_by_min = sorted(results, key=lambda x: x[2], reverse=True)
    best_by_min = results_by_min[0]

    # Print out summary
    print("=== If you KNEW your baseline ===")
    for i, (impr, (move_str, k)) in enumerate(best_per_combo):
        # combos[i] is the baseline
        x1, x2, x3 = combos[i]
        print(
            f"Baseline {i + 1}/{len(combos)}: (x1={x1}, x2={x2}, x3={x3}), "
            f"best single swap: {move_str}, k={k}, improvement={impr:.6f}"
        )

    print("\n=== Single-swap with BEST AVERAGE improvement across all 27 ===")
    print(
        f"Move={best_by_avg[0]}, average_improvement={best_by_avg[1]:.6f},"
        " worst_case_improvement={best_by_avg[2]:.6f}"
    )

    print(
        "\n=== Single-swap with BEST WORST-CASE improvement across all 27 ===",
    )
    print(
        f"Move={best_by_min[0]}, worst_case_improvement={best_by_min[2]:.6f},"
        " average_improvement={best_by_min[1]:.6f}"
    )

    def get_single_swaps_with_k(x1: int, x2: int, x3: int, k: int):
        """
        Yields all possible single swaps (i->j, k) from (x1,x2,x3).
        i->j means move k from x_i to x_j, for k = 1..x_i, i != j.
        We'll yield (move_str, new_x1, new_x2, new_x3).
        """
        original = [x1, x2, x3]
        for i, j in itertools.permutations([0, 1, 2], 2):  # i!=j
            if original[i] >= k:
                newvals = original[:]
                newvals[i] -= k
                newvals[j] += k
                move_str = f"{i + 1}->{j + 1}"
                yield (move_str, newvals[0], newvals[1], newvals[2])

    print(f"Baseline Score: {baseline_sc:.6f}")

    guess_k = [2, 3]
    for idx, (x1, x2, x3) in enumerate(combos):
        print(
            f"\nBaseline {idx + 1}/{len(combos)}: (x1={x1}, x2={x2}, x3={x3})",
        )
        baseline_arr = build_combined_grades(x1, x2, x3)
        baseline_sc = score(baseline_arr)
        print(f"Baseline Score: {baseline_sc:.6f}")

        for k in guess_k:
            print(f"\nFor k = {k}:")
            for move_str, new_x1, new_x2, new_x3 in get_single_swaps_with_k(
                x1, x2, x3, k
            ):
                new_arr = build_combined_grades(new_x1, new_x2, new_x3)
                new_sc = score(new_arr)
                impr = new_sc - baseline_sc
                print(
                    f"Move: {move_str}, New Score: {new_sc:.6f}, "
                    f"Improvement: {impr:.6f}"
                )
