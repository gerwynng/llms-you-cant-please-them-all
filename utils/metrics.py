from typing import Any

from numpy.typing import NDArray


def score(res: NDArray[Any], avg_s: float = 0.2) -> float:
    """
    https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/563931#3128775

    Parameter:
    res: array of shape (n, 3) with entries in the range 0..9
    """
    if res.shape[1] != 3:
        raise ValueError(f"res.shape([1]) == {res.shape[1]}, should be 3")
    if res.mean() == 9:
        return 0
    min_v: float = res.std(axis=0, ddof=1).min()  # minimum of column stds
    avg_h: float = res.std(axis=1, ddof=0).mean()  # average of row stds
    return avg_h * min_v / avg_s / (9 - res.mean())
