import numpy as np

from typing import Tuple


def uniform(
        min_val: float,
        max_val: float,
        dims: Tuple[int, ...],
) -> np.ndarray:
    return (max_val - min_val) * np.random.rand(*dims) + min_val


def normal(
        mean: float,
        std: float,
        dims: Tuple[int],
) -> np.ndarray:
    return std * np.random.rand(*dims) + mean


def generate_random_data(
        num_regions: int = 20,
        num_classes: int = 3,
        num_periods: int = 5,
        min_pop: float = 0.5,
        max_pop: float = 10.,
        min_active_cases_pct: float = 2e-2,
        max_active_cases_pct: float = 5e-2,
        min_closed_cases_pct: float = 5e-2,
        max_closed_cases_pct: float = 2e-1,
        min_rep_factor: float = 1.,
        max_rep_factor: float = 3.,
        min_morbidity_rate: float = 5e-3,
        max_morbidity_rate: float = 3e-2,
        budget_pct: float = 1e-1,
        seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    pop = uniform(
        min_val=min_pop,
        max_val=max_pop,
        dims=(num_regions, num_classes)
    )
    active_cases = pop * uniform(
        min_val=min_active_cases_pct,
        max_val=max_active_cases_pct,
        dims=(num_regions, num_classes)
    )
    closed_cases = pop * uniform(
        min_val=min_closed_cases_pct,
        max_val=max_closed_cases_pct,
        dims=(num_regions, num_classes)
    )
    rep_factor = uniform(
        min_val=min_rep_factor,
        max_val=max_rep_factor,
        dims=(num_regions,)
    )
    morbidity_rate = uniform(
        min_val=min_morbidity_rate,
        max_val=max_morbidity_rate,
        dims=(num_classes,)
    )
    budget = pop.sum() * budget_pct * np.ones(num_periods)
    budget[0] = 0
    immunized_pop = active_cases + closed_cases
    return pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget
