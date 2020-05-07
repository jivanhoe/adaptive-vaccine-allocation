import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def solve_proportional_allocation_model(
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        budget: np.ndarray,
        with_prioritization: bool = True,
        delta: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Define sets
    num_regions = pop.shape[0]
    num_classes = pop.shape[1]
    num_periods = budget.shape[0]
    regions = range(num_regions)
    periods = range(1, num_periods)

    # Define variables
    vaccines = np.zeros((num_regions, num_classes, num_periods))
    cases = np.zeros((num_regions, num_classes, num_periods))
    unimmunized_pop = np.zeros((num_regions, num_classes, num_periods))
    deaths = np.zeros((num_regions, num_classes, num_periods))

    # Set initial conditions
    vaccines[:, :, 0] = 0
    cases[:, :, 0] = active_cases
    unimmunized_pop[:, :, 0] = pop - immunized_pop

    # Calculate dynamics
    for t in periods:
        for i in regions:
            vaccines_available_for_region = pop[i, :].sum() / pop.sum() * budget[t]
            for k in np.argsort(-morbidity_rate[i, :]):
                if with_prioritization:
                    vaccines[i, k, t] = min(unimmunized_pop[i, k, t - 1], vaccines_available_for_region)
                    vaccines_available_for_region -= vaccines[i, k, t]
                else:
                    vaccines[i, k, t] = pop[i, k] / pop[i, :].sum() * vaccines_available_for_region
                cases[i, k, t] = (1 + delta / np.sqrt(t)) ** t * rep_factor[i] / pop[i].sum() * (
                            unimmunized_pop[i, k, t - 1] - vaccines[i, k, t]) * cases[i, :, t - 1].sum()
                unimmunized_pop[i, k, t] = max(unimmunized_pop[i, k, t - 1] - cases[i, k, t] - vaccines[i, k, t], 0)
                deaths[i, k, t] = morbidity_rate[i, k] * cases[i, k, t - 1]

    return vaccines, cases, unimmunized_pop, deaths

