import logging
from typing import Callable, Dict, Tuple, Optional
from models.nominal_model import solve_nominal_model

from copy import deepcopy
from typing import List
import numpy as np
import  gurobipy

logger = logging.getLogger(__name__)


class FoldingHorizonAllocationModel:

    def __init__(
            self,
            solver: Callable = solve_nominal_model,
            solver_params: Optional[Dict[str, any]] = None,
            planning_horizon: int = 5,
            name: Optional[str] = None
     ):
        self.solver = solver
        self.solver_params = solver_params
        self.planning_horizon = planning_horizon
        self.name = name

    @staticmethod
    def update_params(
            vaccines: np.ndarray,
            pop: np.ndarray,
            immunized_pop: np.ndarray,
            active_cases: np.ndarray,
            rep_factor: np.ndarray,
            morbidity_rate: np.ndarray,
            rep_factor_noise: float,
            morbiditiy_rate_noise: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        num_regions, num_classes = immunized_pop.shape

        rep_factor = rep_factor * np.maximum(1 + rep_factor_noise * np.random.randn(num_regions), 0)
        realized_morbidity_rate = morbidity_rate * np.maximum(
            (1 + morbiditiy_rate_noise * np.random.randn(num_regions, num_classes)), 0
        )

        immunized_pop = immunized_pop + vaccines[:, :, 1]
        old_active_cases = active_cases
        for i in range(num_regions):
            active_cases[i, :] = rep_factor[i] / pop[i].sum() * (
                    pop[i, :] - immunized_pop[i, :]) * old_active_cases[i, :].sum()
        immunized_pop = immunized_pop + active_cases

        return immunized_pop, active_cases, rep_factor, realized_morbidity_rate

    def solve_folding_horizon(
            self,
            pop: np.ndarray,
            immunized_pop: np.ndarray,
            active_cases: np.ndarray,
            rep_factor: np.ndarray,
            morbidity_rate: np.ndarray,
            budget: np.ndarray,
            rep_factor_noise: float = 0,
            morbiditiy_rate_noise: float = 0,
            seed: int = 0
    ) -> Tuple[np.array, np.ndarray, np.ndarray, np.ndarray]:

        np.random.seed(seed)

        num_regions, num_classes = immunized_pop.shape
        num_periods = budget.shape[0]

        fh_vaccines = np.zeros((num_regions, num_classes, num_periods))
        fh_immunized_pop = np.zeros((num_regions, num_classes, num_periods))
        fh_cases = np.zeros((num_regions, num_classes, num_periods))
        fh_deaths = np.zeros((num_regions, num_classes, num_periods))

        fh_immunized_pop[:, :, 0] = immunized_pop
        fh_cases[:, :, 0] = active_cases
        fh_deaths[:, :, 0] = active_cases * morbidity_rate

        for t in range(1, num_periods):
            # Solve model
            vaccines, _, _, _ = self.solver(
                pop=pop,
                immunized_pop=immunized_pop,
                active_cases=active_cases,
                rep_factor=rep_factor,
                morbidity_rate=morbidity_rate,
                budget=budget[t - 1:t - 1 + self.planning_horizon],
                **(self.solver_params if self.solver_params else {})
            )

            # Update params
            immunized_pop, active_cases, rep_factor, realized_morbidity_rate = self.update_params(
                vaccines=vaccines,
                pop=pop,
                immunized_pop=immunized_pop,
                active_cases=active_cases,
                rep_factor=rep_factor,
                morbidity_rate=morbidity_rate,
                rep_factor_noise=rep_factor_noise,
                morbiditiy_rate_noise=morbiditiy_rate_noise
            )

            # Store data
            fh_vaccines[:, :, t] = vaccines[:, :, 1]
            fh_immunized_pop[:, :, t] = immunized_pop
            fh_cases[:, :, t] = active_cases
            fh_deaths[:, :, t] = realized_morbidity_rate * active_cases

        return fh_vaccines, fh_immunized_pop, fh_cases, fh_deaths

    def run_simulations(
            self,
            pop: np.ndarray,
            immunized_pop: np.ndarray,
            active_cases: np.ndarray,
            rep_factor: np.ndarray,
            morbidity_rate: np.ndarray,
            budget: np.ndarray,
            rep_factor_noise: float = 0,
            morbidity_rate_noise: float = 0,
            num_trials: int = 5
    ) -> List[Dict[str, any]]:
        results = []
        for seed in range(num_trials):
            try:
                _, _, cases, deaths = self.solve_folding_horizon(
                    pop=deepcopy(pop),
                    immunized_pop=deepcopy(immunized_pop),
                    active_cases=deepcopy(active_cases),
                    rep_factor=deepcopy(rep_factor),
                    morbidity_rate=deepcopy(morbidity_rate),
                    budget=budget,
                    rep_factor_noise=rep_factor_noise,
                    morbiditiy_rate_noise=morbidity_rate_noise,
                    seed=seed
                )
                results.append(
                    dict(
                        total_cases=cases.sum(),
                        total_deaths=deaths.sum(),
                        rep_factor_noise=rep_factor_noise,
                        morbidity_rate_noise=morbidity_rate_noise,
                        model=self.name,
                        **self.solver_params
                    )
                )
            except gurobipy.GurobiError:
                logger.info("Could not find solution for trial.")
        return results
