import numpy as np

from typing import Callable, Dict, Tuple


def update_params(
        vaccines: np.ndarray,
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        noise: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    num_regions, num_classes = immunized_pop.shape
    
    rep_factor = rep_factor * (1 + noise * np.random.randn(num_regions, num_classes))
    morbidity_rate = morbidity_rate * (1 + noise * np.random.randn(num_regions, num_classes))
    
    immunized_pop = immunized_pop - vaccines[:, :, 1]
    old_active_cases = active_cases
    for i in range(num_regions):
        active_cases[i, :] = rep_factor[i, :] / pop[i].sum() * immunized_pop[i, :] * old_active_cases[i, :].sum()
    immunized_pop = immunized_pop - active_cases
    
    return immunized_pop, active_cases, rep_factor, morbidity_rate


def solve_folding_horizon(
        solver: Callable,
        solver_params: Dict[str, any],
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        budget: np.ndarray,
) -> Tuple[np.array, np.ndarray, np.ndarray, np.ndarray]:
    
    num_regions, num_classes = immunized_pop.shape
    num_periods = budget.shape[0]
    
    fh_vaccines = np.zeros((num_regions, num_classes, num_periods))
    fh_unimmunized_pop = np.zeros((num_regions, num_classes, num_periods))
    fh_cases = np.zeros((num_regions, num_classes, num_periods))
    fh_deaths = np.zeros((num_regions, num_classes, num_periods))

    fh_unimmunized_pop[:, :, 0] = immunized_pop
    fh_cases[:, :, 0] = active_cases
    fh_deaths[:, :, 0] = active_cases * morbidity_rate
    
    for t in range(1, num_periods):
        
        # Solve model
        vaccines, _, _, _ = solver(
            pop=pop,
            immunized_pop=immunized_pop,
            active_cases=active_cases,
            rep_factor=rep_factor,
            morbidity_rate=morbidity_rate,
            budget=budget[t - 1:],
            **solver_params
        )
        
        # Update params
        immunized_pop, active_cases, rep_factor, morbidity_rate = update_params(
            vaccines=vaccines,
            pop=pop,
            immunized_pop=immunized_pop,
            active_cases=active_cases,
            rep_factor=rep_factor,
            morbidity_rate=morbidity_rate
        )
        
        # Store data
        fh_vaccines[:, :, t] = vaccines[:, :, 1]
        fh_unimmunized_pop[:, :, t] = immunized_pop
        fh_cases[:, :, t] = active_cases
        fh_deaths[:, :, t] = morbidity_rate * active_cases
        
    return fh_vaccines, fh_unimmunized_pop, fh_cases, fh_deaths
