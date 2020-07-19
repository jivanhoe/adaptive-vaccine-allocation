import numpy as np
import gurobipy as gp
from gurobipy import GRB

from typing import Optional, Tuple


class MortalityRateEstimator:

    def __init__(
            self,
            baseline_mortality_rate: np.ndarray,
            deaths: np.ndarray,
            cases: np.ndarray,
            population: np.ndarray,
            n_timesteps_per_estimate: int,
            max_pct_change: float,
            max_pct_population_deviation: float

    ):
        # Set provided attributes
        self.baseline_mortality_rate = baseline_mortality_rate
        self.deaths = deaths
        self.cases = cases
        self.population = population
        self.n_timesteps_per_estimate = n_timesteps_per_estimate
        self.max_pct_change = max_pct_change
        self.max_pct_population_deviation = max_pct_population_deviation

        # Set helper attributes
        self._pop_proportions = self.population / self.population.sum()
        self._n_risk_classes = baseline_mortality_rate.shape[0]
        self._n_timesteps = deaths.shape[0]
        self._n_estimation_periods = int(np.ceil(self._n_timesteps / self.n_timesteps_per_estimate))
        self._risk_classes = np.arange(self._n_risk_classes)
        self._timesteps = np.arange(self._n_timesteps)
        self._estimation_periods = np.arange(self._n_estimation_periods)

        self._total_cases = [
            self.cases[p * self.n_timesteps_per_estimate:(p + 1) * self.n_timesteps_per_estimate].sum()
            for p in self._estimation_periods
        ]
        self._total_deaths = [
            self.deaths[p * self.n_timesteps_per_estimate:(p + 1) * self.n_timesteps_per_estimate].sum()
            for p in self._estimation_periods
        ]

    def _get_warm_start(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        deaths = np.zeros((self._n_risk_classes, self._n_estimation_periods))
        cases = np.zeros((self._n_risk_classes, self._n_estimation_periods))
        for p in self._estimation_periods:
            deaths[:, p] = self._total_deaths[p] * self.baseline_mortality_rate / self.baseline_mortality_rate.sum()
            cases[:, p] = self._total_cases[p] * self._pop_proportions
        mortality_rate = deaths / cases
        return mortality_rate, deaths, cases

    def solve(
            self,
            mip_gap: Optional[float] = None,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            output_flag: bool = True
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        # Initialize model
        model = gp.Model()

        # Define decision variables
        mortality_rate = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        deaths = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        cases = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        error = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)

        # Set constraints to enforce consistency with data
        model.addConstrs(
            cases.sum("*", p) == self._total_cases[p]
            for p in self._estimation_periods
        )
        model.addConstrs(
            deaths.sum("*", p) == self._total_deaths[p]
            for p in self._estimation_periods
        )

        # Set constraints to roughly align cases with population subset size
        model.addConstrs(
            cases[k, p] <= (1 + self.max_pct_population_deviation) * self._pop_proportions[k] * self._total_cases[p]
            for k in self._risk_classes for p in self._estimation_periods
        )
        model.addConstrs(
            cases[k, p] <= (1 - self.max_pct_population_deviation) * self._pop_proportions[k] * self._total_cases[p]
            for k in self._risk_classes for p in self._estimation_periods
        )

        # Set constraints for smoothness
        model.addConstrs(
            mortality_rate[k, p + 1] <= (1 + self.max_pct_change) * mortality_rate[k, p]
            for k in self._risk_classes for p in np.arange(self._n_estimation_periods - 1)
        )
        model.addConstrs(
            mortality_rate[k, p + 1] >= (1 - self.max_pct_change) * mortality_rate[k, p]
            for k in self._risk_classes for p in np.arange(self._n_estimation_periods - 1)
        )

        # Set bi-linear constraints to define mortality rate
        model.addConstrs(
            mortality_rate[k, p] * cases[k, p] == deaths[k, p]
            for k in self._risk_classes for p in self._estimation_periods
        )

        # Set objective
        model.addConstrs(
            error[k, p] >= (mortality_rate[k, p] - self.baseline_mortality_rate[k])
            * (mortality_rate[k, p] - self.baseline_mortality_rate[k])
            for k in self._risk_classes for p in self._estimation_periods
        )
        model.setObjective(error.sum(), GRB.MINIMIZE)

        # Set warm start
        _, cases_warm_start, deaths_warm_start = self._get_warm_start()
        for k in self._risk_classes:
            for p in self._estimation_periods:
                cases[k, p].start = cases_warm_start[k, p]
                deaths[k, p].start = deaths_warm_start[k, p]

        # Set model params
        if mip_gap:
            model.params.MIPGap = mip_gap
        if feasibility_tol:
            model.params.FeasibilityTol = feasibility_tol
        if time_limit:
            model.params.TimeLimit = time_limit
        model.params.OutputFlag = output_flag
        model.params.NonConvex = 2

        # Solve model
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return np.ndarray(model.getAttr("x", mortality_rate)), \
                   np.ndarray(model.getAttr("x", deaths)), \
                   np.ndarray(model.getAttr("x", cases))

        print('\nModel was not solved to optimality.')
