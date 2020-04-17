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
        budget: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Define sets
    num_regions = pop.shape[0]
    num_classes = pop.shape[1]
    num_periods = budget.shape[0]
    regions = range(num_regions)
    risk_classes = range(num_classes)
    periods = range(1, num_periods)
    logger.debug(f"Regions: {num_regions} \t Risk classes: {num_classes} \t Periods: {num_periods}")

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
    for i in regions:
        for k in risk_classes:
            for t in periods:
                vaccines[i, k, t] = pop[i, k] / pop.sum() * budget[t]
                cases[i, k, t] = rep_factor[i] / pop[i].sum() * unimmunized_pop[i, k, t - 1] * cases[i, :, t-1].sum()
                unimmunized_pop[i, k, t] = max(unimmunized_pop[i, k, t - 1] - cases[i, k, t] - vaccines[i, k, t], 0)
                deaths[i, k, t] = morbidity_rate[k] * cases[i, k, t-1]

    return vaccines, cases, unimmunized_pop, deaths
