import numpy as np
import pandas as pd
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
            objective_weight: Optional[float] = 1e-3,
            mip_gap: Optional[float] = None,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            output_flag: bool = True,
            check_model_output: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        # Initialize model
        model = gp.Model()

        # Define decision variables
        mortality_rate = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        deaths = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        cases = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        mortality_error = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        cases_error = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        cases_bound = model.addVar()
        deaths_bound = model.addVar()

        # Set constraints to enforce consistency with data
        model.addConstrs(
             cases_bound >= self._total_cases[p] - cases.sum("*", p)
             for p in self._estimation_periods
        )
        model.addConstrs(
            cases_bound >= - self._total_cases[p] + cases.sum("*", p)
            for p in self._estimation_periods
        )
        model.addConstrs(
            deaths_bound >= self._total_deaths[p] - deaths.sum("*", p)
            for p in self._estimation_periods
        )
        model.addConstrs(
            deaths_bound >= - self._total_deaths[p] + deaths.sum("*", p)
            for p in self._estimation_periods
        )

        # Set constraints to roughly align cases with population subset size
        model.addConstrs(
            cases_error[k, p] >= cases[k, p] - self._pop_proportions[k] * self._total_cases[p]
            for k in self._risk_classes for p in self._estimation_periods
        )
        model.addConstrs(
            cases_error[k, p] >= self._pop_proportions[k] * self._total_cases[p] - cases[k, p]
            for k in self._risk_classes for p in self._estimation_periods
        )

        # Set feasibility constraint
        model.addConstrs(cases[k, p] >= deaths[k, p] for k in self._risk_classes for p in self._estimation_periods)

        # Set bi-linear constraints to define mortality rate
        model.addConstrs(
            mortality_rate[k, p] * cases[k, p] == deaths[k, p]
            for k in self._risk_classes for p in self._estimation_periods
        )

        # Set objective
        model.addConstrs(
            mortality_error[k, p] >= (mortality_rate[k, p] - self.baseline_mortality_rate[k])
            for k in self._risk_classes for p in self._estimation_periods
        )
        model.addConstrs(
            mortality_error[k, p] >= - (mortality_rate[k, p] + self.baseline_mortality_rate[k])
            for k in self._risk_classes for p in self._estimation_periods
        )
        model.setObjective(mortality_error.sum() + objective_weight * (cases_error.sum() + cases_bound + deaths_bound)
                           , GRB.MINIMIZE)

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

        # Infeasibility de-bug
        if model.status == GRB.INFEASIBLE:
            model.computeIIS()
            print('\nModel was not solved to optimality.')
            print(f"Violated constraints at indices: {[idx for idx, val in enumerate(model.IISCONSTR) if val == 1]}")

        if model.status == GRB.OPTIMAL:
            mortality_rate = model.getAttr("x", mortality_rate)
            deaths = model.getAttr("x", deaths)
            cases = model.getAttr("x", cases)

            mortality_rate = np.array([
                [mortality_rate[k, t] for t in self._estimation_periods] for k in self._risk_classes
            ])
            deaths = np.array([
                [deaths[k, t] for t in self._estimation_periods] for k in self._risk_classes
            ])
            cases = np.array([
                [cases[k, t] for t in self._estimation_periods] for k in self._risk_classes
            ])

            if check_model_output:
                print("\n" * 2, self._summarize_mortality_estimates(mortality_rate, deaths, cases))

            return mortality_rate, deaths, cases

    def _summarize_mortality_estimates(self,
                                       mortality_rate: np.ndarray,
                                       deaths: np.ndarray,
                                       cases: np.ndarray) -> pd.DataFrame:

        table = pd.DataFrame({
            'Baseline Mortality': self.baseline_mortality_rate,
            'Model Mortality': mortality_rate.mean(axis=1),
            'True Cases (avg)': self._pop_proportions * np.mean(self._total_cases),
            'Model Cases (avg)': cases.mean(axis=1),
            'True Deaths (avg)': self._pop_proportions * np.mean(self._total_deaths),
            'Model Deaths (avg)': deaths.mean(axis=1)

        }, index=['0-9', '10-49', '50-59', '60-69', '70-79', '80-inf'])

        return table
