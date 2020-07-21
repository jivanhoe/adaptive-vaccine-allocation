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
            max_pct_mortality_rate_deviation: float,
            max_pct_cases_deviation: float,
            regularization_param: float,
            use_l2_error: bool,
            relax_cases_deviation_constraint: bool
    ):
        # Set provided attributes
        self.baseline_mortality_rate = baseline_mortality_rate
        self.deaths = deaths
        self.cases = cases
        self.population = population
        self.n_timesteps_per_estimate = n_timesteps_per_estimate
        self.max_pct_change = max_pct_change
        self.max_pct_mortality_rate_deviation = max_pct_mortality_rate_deviation
        self.max_pct_cases_deviation = max_pct_cases_deviation
        self.regularization_param = regularization_param
        self.use_l2_error = use_l2_error
        self.relax_cases_deviation_constraint = relax_cases_deviation_constraint

        # Set helper attributes
        self._n_risk_classes = baseline_mortality_rate.shape[0]
        self._n_timesteps = deaths.shape[0]
        self._n_estimation_periods = int(np.ceil(self._n_timesteps / self.n_timesteps_per_estimate))
        self._risk_classes = np.arange(self._n_risk_classes)
        self._timesteps = np.arange(self._n_timesteps)
        self._estimation_periods = np.arange(self._n_estimation_periods)
        self._pop_proportions = self.population / self.population.sum()
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

    def _process_solution(
            self,
            mortality_rate: np.ndarray,
            deaths: np.ndarray,
            cases: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dims = (self._n_risk_classes, self._n_timesteps)
        resampled_mortality_rates = np.zeros(dims)
        resampled_deaths = np.zeros(dims)
        resampled_cases = np.zeros(dims)
        for p in self._estimation_periods:
            start = p * self.n_timesteps_per_estimate
            end = min((p + 1) * self.n_timesteps_per_estimate, self._n_timesteps)
            for t in range(start, end):
                resampled_mortality_rates[:, t] = mortality_rate[:, p]
                resampled_deaths[:, t] = deaths[:, p] * self.deaths[t] / self.deaths[start:end].sum()
                resampled_cases[:, t] = cases[:, p] * self.cases[t] / self.cases[start:end].sum()
        return resampled_mortality_rates, resampled_deaths, resampled_cases

    def solve(
            self,
            mip_gap: Optional[float] = None,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            output_flag: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        # Initialize model
        model = gp.Model()

        # Define decision variables
        mortality_rate = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0, ub=1)
        deaths = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        cases = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        mortality_rate_deviation = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)
        cases_deviation = model.addVars(self._n_risk_classes, self._n_estimation_periods, lb=0)

        # Set bi-linear constraints to define mortality rate
        model.addConstrs(
            (
                mortality_rate[k, p] * cases[k, p] == deaths[k, p]
                for k in self._risk_classes for p in self._estimation_periods
            ),
            name="Mortality rate definition"
        )

        # Set constraints to enforce consistency with data
        model.addConstrs(
            (
                cases.sum("*", p) == self._total_cases[p]
                for p in self._estimation_periods
            ),
            name="Total cases sum"
        )
        model.addConstrs(
            (
                deaths.sum("*", p) == self._total_deaths[p]
                for p in self._estimation_periods
            ),
            name="Total deaths sum"
        )

        # Set smoothness constraints
        model.addConstrs(
            (
                mortality_rate[k, p + 1] <= (1 + self.max_pct_change) * mortality_rate[k, p]
                for k in self._risk_classes for p in np.arange(self._n_estimation_periods - 1)
            ),
            name="Mortality rate smoothness upper bound"
        )
        model.addConstrs(
            (
                mortality_rate[k, p + 1] >= (1 - self.max_pct_change) * mortality_rate[k, p]
                for k in self._risk_classes for p in np.arange(self._n_estimation_periods - 1)
            ),
            name="Mortality rate smoothness lower bound"
        )

        # Set mortality rate ordering constraints
        for k in self._risk_classes:
            for l in self._risk_classes:
                if self.baseline_mortality_rate[k] > self.baseline_mortality_rate[l]:
                    model.addConstrs(
                        (
                            mortality_rate[k, p] >= mortality_rate[l, p]
                            for p in self._estimation_periods
                        ),
                        name=f"Mortality rate ordering between classes {k} and {l}"
                    )

        # Set constraints to roughly align cases with population subset size
        model.addConstrs(
            cases_deviation[k, p] >= cases[k, p] - self._pop_proportions[k] * self._total_cases[p]
            for k in self._risk_classes for p in self._estimation_periods
        )
        model.addConstrs(
            cases_deviation[k, p] >= self._pop_proportions[k] * self._total_cases[p] - cases[k, p]
            for k in self._risk_classes for p in self._estimation_periods
        )
        if not self.relax_cases_deviation_constraint:
            model.addConstrs(
                (
                    cases_deviation[k, p] <= self.max_pct_cases_deviation * self._pop_proportions[k] * self._total_cases[p]
                    for k in self._risk_classes for p in self._estimation_periods
                ),
                name="Cases deviation upper bound"
            )

        # Set mortality rate error constraints
        model.addConstrs(
            (
                mortality_rate_deviation[k, p] >= mortality_rate[k, p] - self.baseline_mortality_rate[k]
                for k in self._risk_classes for p in self._estimation_periods
            ),
            name="Error lower bound 1"
        )
        model.addConstrs(
            (
                mortality_rate_deviation[k, p] >= self.baseline_mortality_rate[k] - mortality_rate[k, p]
                for k in self._risk_classes for p in self._estimation_periods
            ),
            name="Error lower bound 2"
        )
        model.addConstrs(
            (
                mortality_rate_deviation[k, p] <= self.max_pct_mortality_rate_deviation * self.baseline_mortality_rate[k]
                for k in self._risk_classes for p in self._estimation_periods
            ),
            name="Error upper bound"
        )

        # Set objective
        if self.use_l2_error:
            loss = gp.quicksum(
                mortality_rate_deviation[k, p] * mortality_rate_deviation[k, p]
                for k in self._risk_classes for p in self._estimation_periods
            )
        else:
            loss = mortality_rate_deviation.sum()
        if self.relax_cases_deviation_constraint:
            loss = loss + self.regularization_param * cases_deviation.sum()
        model.setObjective(loss, GRB.MINIMIZE)

        # Set warm start
        mortality_rate_start, deaths_warm_start, cases_warm_start = self._get_warm_start()
        for k in self._risk_classes:
            for p in self._estimation_periods:
                mortality_rate.start = mortality_rate_start[k, p]
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

        # If infeasible, compute IIS for debugging
        if model.status == GRB.INFEASIBLE:
            model.computeIIS()
            print(f"Violated constraints at indices: {[idx for idx, val in enumerate(model.IISCONSTR) if val == 1]}")
            return

        # Return solution as arrays
        to_array = lambda x: np.array([[x[k, p] for p in self._estimation_periods] for k in self._risk_classes])
        mortality_rate = to_array(model.getAttr("x", mortality_rate))
        deaths = to_array(model.getAttr("x", deaths))
        cases = to_array(model.getAttr("x", cases))
        return self._process_solution(
            mortality_rate=mortality_rate,
            deaths=deaths,
            cases=cases
        )

    def _summarize_mortality_estimates(self,
                                       mortality_rate: np.ndarray,
                                       deaths: np.ndarray,
                                       cases: np.ndarray) -> pd.DataFrame:

        table = pd.DataFrame({
            'Baseline mortality': self.baseline_mortality_rate,
            'Estimated mortality': mortality_rate.mean(axis=1),
            'Expected total cases (avg)': self._pop_proportions * np.mean(self._total_cases),
            'Model Cases (avg)': cases.mean(axis=1),
            'True Deaths (avg)': self._pop_proportions * np.mean(self._total_deaths),
            'Model Deaths (avg)': deaths.mean(axis=1)

        }, index=['0-9', '10-49', '50-59', '60-69', '70-79', '80-inf'])

        return table
