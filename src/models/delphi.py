from typing import Tuple, Union, Optional, NoReturn, List, Dict

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from gurobipy import GurobiError


class DiscreteDELPHISolution:

    def __init__(
            self,
            susceptible: np.ndarray,
            exposed: np.ndarray,
            infectious: np.ndarray,
            hospitalized_dying: np.ndarray,
            hospitalized_recovering: np.ndarray,
            quarantined_dying: np.ndarray,
            quarantined_recovering: np.ndarray,
            undetected_dying: np.ndarray,
            undetected_recovering: np.ndarray,
            deceased: np.ndarray,
            recovered: np.ndarray,
            vaccinated: np.ndarray,
            population: np.ndarray,
            capacity: Optional[np.ndarray] = None,
            days_per_timestep: float = 1.0,
            vaccine_effectiveness: float = 1.0,
            validate_on_init: bool = False
    ):
        """
        Instantiate a container object for a DELPHI solution
        :param susceptible: a numpy array of shape (n_regions, k_classes, t_timesteps + 1) that represents the
            susceptible population by region and risk class at each timestep
        :param exposed: a numpy array of shape (n_regions, k_classes, t_timesteps + 1) that represents the exposed
            population by region and risk class at each timestep
        :param infectious: a numpy array of shape [n_regions, k_classes, t_timesteps + 1) that represents the infectious
            population by region and risk class at each timestep
        :param hospitalized_dying: a numpy array of shape (n_regions, k_classes, t_timesteps + 1) that represents the
            hospitalized dying population by region and risk class at each timestep
        :param hospitalized_recovering: a numpy array of shape (n_regions, k_classes, t_timesteps + 1) that represents
            the hospitalized recovering population by region and risk class at each timestep
        :param quarantined_dying: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the
            quarantined dying population by region and risk class at each timestep
        :param quarantined_recovering: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the
            quarantined recovering population by region and risk class at each timestep
        :param undetected_dying: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the
            undetected dying population by region and risk class at each timestep
        :param undetected_recovering: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the
            undetected recovering population by region and risk class at each timestep
        :param recovered: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the recovered
            population by region and risk class at each timestep
        :param deceased: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the deceased
            population by region and risk class at each timestep
        :param vaccinated: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the number
            vaccines allocated by region and risk class at each timestep
        :param population: a numpy array of shape (
        :param validate_on_init: a boolean that specifies whether to validate the provided solution on initialization
        """

        # Initialize functional states
        self.susceptible = susceptible
        self.exposed = exposed
        self.infectious = infectious
        self.hospitalized_dying = hospitalized_dying
        self.hospitalized_recovering = hospitalized_recovering
        self.hospitalized = hospitalized_dying + hospitalized_recovering
        self.quarantined_dying = quarantined_dying
        self.quarantined_recovering = quarantined_recovering
        self.quarantined = quarantined_dying + quarantined_recovering
        self.undetected_dying = undetected_dying
        self.undetected_recovering = undetected_recovering
        self.undetected = undetected_dying + undetected_recovering
        self.deceased = deceased
        self.recovered = recovered
        self.vaccinated = vaccinated
        self.population = population
        self.capacity = capacity
        self.vaccine_effectiveness = vaccine_effectiveness
        self.days_per_timestep = days_per_timestep

        # Check solution
        if validate_on_init:
            self._validate_solution()

    def _validate_solution(self) -> NoReturn:
        """
        Check that the provided solution arrays have valid dimensions and values.
        :return: None
        """
        expected_dims = self.susceptible.shape
        for attr, value in self.__dict__.items():
            if attr not in ["days_per_timestep", "vaccine_effectiveness", "population", "capacity"] and value:
                assert value.shape == expected_dims, \
                    f"Invalid dimensions for {attr} array - expected {expected_dims}, received {value.shape}"
                assert np.all(value >= 0), f"Invalid {attr} array - all values must be non-negative"

    def get_total_deaths(self) -> float:
        return self.deceased[:, :, -1].sum()

    def get_objective(self) -> float:
        return (self.deceased + self.hospitalized_dying + self.quarantined_dying + self.undetected_dying
                )[:, :, -2].sum()

    def get_total_infections(self) -> float:
        infectious = self.infectious.sum(axis=(0, 1))
        return 0.5 * (infectious[1:] + infectious[:-1]).sum() * self.days_per_timestep


class PrescriptiveDELPHIModel:

    def __init__(
            self,
            initial_conditions: Dict[str, np.ndarray],
            delphi_params: Dict[str, Union[float, np.ndarray]],
            vaccine_params: Dict[str, Union[float, bool, np.ndarray]]
    ):
        """
        Instantiate a discrete DELPHI model with initial conditions and parameter estimates.
        :param initial_conditions: a dictionary containing initial condition arrays
        :param delphi_params: a dictionary containing the estimate model parameters
        """

        # Set initial conditions
        self.initial_susceptible = initial_conditions["initial_susceptible"]
        self.initial_exposed = initial_conditions["initial_exposed"]
        self.initial_infectious = initial_conditions["initial_infectious"]
        self.initial_hospitalized_dying = initial_conditions["initial_hospitalized_dying"]
        self.initial_hospitalized_recovering = initial_conditions["initial_hospitalized_recovering"]
        self.initial_quarantined_dying = initial_conditions["initial_quarantined_dying"]
        self.initial_quarantined_recovering = initial_conditions["initial_quarantined_recovering"]
        self.initial_undetected_dying = initial_conditions["initial_undetected_dying"]
        self.initial_undetected_recovering = initial_conditions["initial_undetected_recovering"]
        self.initial_recovered = initial_conditions["initial_recovered"]
        self.population = initial_conditions["population"]

        # Set DELPHI model parameters
        self.infection_rate = delphi_params["infection_rate"]
        self.policy_response = delphi_params["policy_response"]
        self.progression_rate = delphi_params["progression_rate"]
        self.detection_rate = delphi_params["detection_rate"]
        self.ihd_transition_rate = delphi_params["ihd_transition_rate"]
        self.ihr_transition_rate = delphi_params["ihr_transition_rate"]
        self.iqd_transition_rate = delphi_params["iqd_transition_rate"]
        self.iqr_transition_rate = delphi_params["iqr_transition_rate"]
        self.iud_transition_rate = delphi_params["iud_transition_rate"]
        self.iur_transition_rate = delphi_params["iur_transition_rate"]
        self.death_rate = delphi_params["death_rate"]
        self.recovery_rate = delphi_params["recovery_rate"]
        self.mortality_rate = delphi_params["mortality_rate"]
        self.days_per_timestep = delphi_params["days_per_timestep"]

        # Set vaccine parameters
        self.vaccine_budget = vaccine_params["vaccine_budget"]
        self.max_allocation_pct = vaccine_params["max_allocation_pct"]
        self.min_allocation_pct = vaccine_params["min_allocation_pct"]
        self.max_total_capacity = vaccine_params["max_total_capacity"]
        self.optimize_capacity = vaccine_params["optimize_capacity"]
        self.vaccine_effectiveness = vaccine_params["vaccine_effectiveness"]
        self.excluded_risk_classes = vaccine_params["excluded_risk_classes"]

        # Initialize helper attributes
        self._n_regions = self.initial_susceptible.shape[0]
        self._n_risk_classes = self.initial_susceptible.shape[1]
        self._n_included_risk_classes = self._n_risk_classes - self.excluded_risk_classes.shape[0]
        self._n_timesteps = self.vaccine_budget.shape[0]
        self._regions = np.arange(self._n_regions)
        self._risk_classes = np.arange(self._n_risk_classes)
        self._included_risk_classes = np.array([k for k in self._risk_classes if k not in self.excluded_risk_classes])
        self._timesteps = np.arange(self._n_timesteps)
        self._planning_timesteps = np.array([t for t in self._timesteps if self.vaccine_budget[t] > 0])
        self._non_planning_timesteps = np.array([t for t in self._timesteps if self.vaccine_budget[t] == 0])

    def simulate(
            self,
            vaccinated: Optional[np.ndarray] = None,
            randomize_allocation: bool = False,
            prioritization_allocation: bool = False
    ) -> DiscreteDELPHISolution:
        """
        Solve DELPHI system using a forward difference scheme.
        :param vaccinated: a numpy array of (n_regions, n_classes, n_timesteps + 1) that represents a feasible
            allocation of vaccines by region and risk class at each timestep
        :param randomize_allocation: a boolean that specifies whether to randomize the allocation policy if none
            provided (default False)
        :param prioritization_allocation: a boolean that specifies whether to prioritize allocation to high risk
            individuals within each region if no allocation policy is provided (default False)
        :return: a DiscreteDELPHISolution object
        """

        # Initialize functional states
        dims = (self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        susceptible = np.zeros(dims)
        exposed = np.zeros(dims)
        infectious = np.zeros(dims)
        hospitalized_dying = np.zeros(dims)
        hospitalized_recovering = np.zeros(dims)
        quarantined_dying = np.zeros(dims)
        quarantined_recovering = np.zeros(dims)
        undetected_dying = np.zeros(dims)
        undetected_recovering = np.zeros(dims)
        deceased = np.zeros(dims)
        recovered = np.zeros(dims)

        # Initialize control variable if none provided
        allocate_vaccines = vaccinated is None
        if allocate_vaccines:
            vaccinated = np.zeros(dims)

        # Set initial conditions
        susceptible[:, :, 0] = self.initial_susceptible
        exposed[:, :, 0] = self.initial_exposed
        infectious[:, :, 0] = self.initial_infectious
        hospitalized_dying[:, :, 0] = self.initial_hospitalized_dying
        hospitalized_recovering[:, :, 0] = self.initial_hospitalized_recovering
        quarantined_dying[:, :, 0] = self.initial_quarantined_dying
        quarantined_recovering[:, :, 0] = self.initial_quarantined_recovering
        undetected_dying[:, :, 0] = self.initial_undetected_dying
        undetected_recovering[:, :, 0] = self.initial_undetected_recovering
        recovered[:, :, 0] = self.initial_recovered

        # Propagate discrete DELPHI dynamics with vaccine allocation heuristic
        for t in self._timesteps:

            # Get eligible population subsets for vaccination
            eligible = susceptible[:, self._included_risk_classes, t] \
                       - (1 - self.vaccine_effectiveness) * vaccinated[:, self._included_risk_classes, :t].sum()
            eligible = np.maximum(eligible, 0)
            total_eligible = eligible.sum()

            if total_eligible > 0 and allocate_vaccines:

                # If random allocation specified, generate feasible allocation
                if randomize_allocation:
                    min_vaccinated = self.min_allocation_pct * eligible
                    additional_budget = self.vaccine_budget[t] - min_vaccinated.sum()
                    additional_proportion = np.random.exponential(size=(self._n_regions, self._n_included_risk_classes)) ** 2
                    additional_proportion = additional_proportion / additional_proportion.sum()
                    vaccinated[:, self._included_risk_classes, t] = np.minimum(
                        eligible,
                        np.minimum(
                            min_vaccinated + additional_proportion * additional_budget,
                            self.max_allocation_pct * self.population[:, self._included_risk_classes]
                        )
                    )

                # Else use baseline policy that orders region-wise allocation by risk class
                else:
                    if prioritization_allocation:
                        regional_budget = eligible.sum(axis=1) / total_eligible * self.vaccine_budget[t]
                        for k in np.argsort(-self.ihd_transition_rate):
                            if k in self._included_risk_classes:
                                vaccinated[:, k, t] = np.minimum(
                                    regional_budget, eligible[:, self._included_risk_classes == k][0])
                                regional_budget -= vaccinated[:, k, t]
                    else:
                        regional_budget = self.population.sum(axis=1) / self.population.sum()
                        for k in self._included_risk_classes:
                            vaccinated[:, k, t] = regional_budget * self.population[:, k] / \
                                                  self.population[:, self._included_risk_classes].sum()

            # Apply Euler forward difference scheme with clipping of negative values
            for j in self._regions:
                susceptible[j, :, t + 1] = susceptible[j, :, t] - self.vaccine_effectiveness * vaccinated[j, :, t] - (
                        self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum()
                        * (susceptible[j, :, t] - self.vaccine_effectiveness * vaccinated[j, :, t])
                        * infectious[j, :, t].sum()
                ) * self.days_per_timestep
                exposed[j, :, t + 1] = exposed[j, :, t] + (
                        self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum()
                        * (susceptible[j, :, t] - self.vaccine_effectiveness * vaccinated[j, :, t]) *
                        infectious[j, :, t].sum()
                        - self.progression_rate * exposed[j, :, t]
                ) * self.days_per_timestep
            susceptible[:, :, t + 1] = np.maximum(susceptible[:, :, t + 1], 0)
            exposed[:, :, t + 1] = np.maximum(exposed[:, :, t + 1], 0)

            infectious[:, :, t + 1] = infectious[:, :, t] + (
                    self.progression_rate * exposed[:, :, t]
                    - self.detection_rate * infectious[:, :, t]
            ) * self.days_per_timestep
            infectious[:, :, t + 1] = np.maximum(infectious[:, :, t + 1], 0)

            hospitalized_dying[:, :, t + 1] = hospitalized_dying[:, :, t] + (
                    self.ihd_transition_rate * infectious[:, :, t]
                    - self.death_rate * hospitalized_dying[:, :, t]
            ) * self.days_per_timestep
            hospitalized_dying[:, :, t + 1] = np.maximum(hospitalized_dying[:, :, t + 1], 0)

            hospitalized_recovering[:, :, t + 1] = hospitalized_recovering[:, :, t] + (
                    self.ihr_transition_rate * infectious[:, :, t]
                    - self.recovery_rate * hospitalized_recovering[:, :, t]
            ) * self.days_per_timestep
            hospitalized_recovering[:, :, t + 1] = np.maximum(hospitalized_recovering[:, :, t + 1], 0)

            quarantined_dying[:, :, t + 1] = quarantined_dying[:, :, t] + (
                    self.iqd_transition_rate * infectious[:, :, t]
                    - self.death_rate * quarantined_dying[:, :, t]
            ) * self.days_per_timestep
            quarantined_dying[:, :, t + 1] = np.maximum(quarantined_dying[:, :, t + 1], 0)

            quarantined_recovering[:, :, t + 1] = quarantined_recovering[:, :, t] + (
                    self.iqr_transition_rate * infectious[:, :, t]
                    - self.recovery_rate * quarantined_recovering[:, :, t]
            ) * self.days_per_timestep
            quarantined_recovering[:, :, t + 1] = np.maximum(quarantined_recovering[:, :, t + 1], 0)

            undetected_dying[:, :, t + 1] = undetected_dying[:, :, t] + (
                    self.iud_transition_rate * infectious[:, :, t]
                    - self.death_rate * undetected_dying[:, :, t]
            ) * self.days_per_timestep
            undetected_dying[:, :, t + 1] = np.maximum(undetected_dying[:, :, t + 1], 0)

            undetected_recovering[:, :, t + 1] = undetected_recovering[:, :, t] + (
                    self.iur_transition_rate * infectious[:, :, t]
                    - self.recovery_rate * undetected_recovering[:, :, t]
            ) * self.days_per_timestep
            undetected_recovering[:, :, t + 1] = np.maximum(undetected_recovering[:, :, t + 1], 0)

            deceased[:, :, t + 1] = deceased[:, :, t] + self.death_rate * (
                    hospitalized_dying[:, :, t] + quarantined_dying[:, :, t] + undetected_dying[:, :, t]
            ) * self.days_per_timestep

            recovered[:, :, t + 1] = recovered[:, :, t] + self.recovery_rate * (
                    hospitalized_recovering[:, :, t] + quarantined_recovering[:, :, t]
                    + undetected_recovering[:, :, t]
            ) * self.days_per_timestep

        return DiscreteDELPHISolution(
            susceptible=susceptible,
            exposed=exposed,
            infectious=infectious,
            hospitalized_dying=hospitalized_dying,
            hospitalized_recovering=hospitalized_recovering,
            quarantined_dying=quarantined_dying,
            quarantined_recovering=quarantined_recovering,
            undetected_dying=undetected_dying,
            undetected_recovering=undetected_recovering,
            recovered=recovered,
            deceased=deceased,
            vaccinated=vaccinated,
            population=self.population,
            days_per_timestep=self.days_per_timestep,
            vaccine_effectiveness=self.vaccine_effectiveness
        )

    def _optimize_relaxation(
            self,
            estimated_infectious: np.ndarray,
            exploration_rel_tol: float,
            exploration_abs_tol: float,
            mip_gap: Optional[float],
            feasibility_tol: Optional[float],
            time_limit: Optional[float],
            disable_crossover: bool,
            output_flag: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the linear relaxation of the  for the optimal vaccine policy, given a fixed infectious pop
        :param estimated_infectious: numpy array of size (n_regions, k_classes, t_timesteps + 1) represents the
            estimated infectious population by region and risk class at each timestep, based on a previous feasible
            solution
        :param exploration_rel_tol: a float in [0, 1] that specifies maximum allowed relative error between the
            estimated  and actual infectious population in any region
        :param exploration_abs_tol: a positive float that specifies maximum allowed absolute error between the
            estimated and actual infectious population in any region
        :param mip_gap: a float that specifies the maximum MIP gap required for termination
        :param feasibility_tol: a float that specifies that maximum feasibility tolerance for constraints
        :param output_flag: a boolean that specifies whether to show the solver logs
        :param disable_crossover: a boolean that if true disables Gurobi's crossover algorithm, which used to clean up
            the interior solution of the barrier method into a basic feasible solution
        :param time_limit: a float that specifies the maximum solve time in seconds
        :return: a numpy array of size (n_regions, k_classes, t_timesteps + 1) that represents the updated vaccine
            allocations
        """

        # Initialize model
        solver = gp.Model("DELPHI")

        # Define decision variables
        susceptible = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        exposed = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        infectious = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        hospitalized_dying = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        quarantined_dying = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        undetected_dying = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        deceased = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        vaccinated = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        infectious_error = solver.addVars(self._n_regions, self._n_timesteps, lb=0)
        surplus_vaccines = solver.addVars(self._n_regions, self._n_risk_classes, lb=0)
        unallocated_vaccines = solver.addVars(self._n_timesteps, lb=0)
        if self.optimize_capacity:
            capacity = solver.addVars(self._n_regions, lb=0)

        # Set initial conditions for DELPHI model
        solver.addConstrs(
            susceptible[j, k, 0] == self.initial_susceptible[j, k]
            for j in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            exposed[j, k, 0] == self.initial_exposed[j, k]
            for j in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            infectious[j, k, 0] == self.initial_infectious[j, k]
            for j in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            hospitalized_dying[j, k, 0] == self.initial_hospitalized_dying[j, k]
            for j in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            quarantined_dying[j, k, 0] == self.initial_quarantined_dying[j, k]
            for j in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            undetected_dying[j, k, 0] == self.initial_undetected_dying[j, k]
            for j in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            deceased[j, k, 0] == 0
            for j in self._regions for k in self._risk_classes
        )

        # Set DELPHI dynamics constraints
        solver.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] + self.vaccine_effectiveness * vaccinated[j, k, t] >=
            - self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum()
            * (susceptible[j, k, t] - self.vaccine_effectiveness * vaccinated[j, k, t])
            * (1 - exploration_rel_tol) * estimated_infectious[j, t] * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] >=
            - self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum() * susceptible[j, k, t]
            * (1 - exploration_rel_tol) * estimated_infectious[j, t] * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum()
                    * (susceptible[j, k, t] - self.vaccine_effectiveness * vaccinated[j, k, t])
                    * (1 + exploration_rel_tol) * estimated_infectious[j, t]
                    - self.progression_rate * exposed[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum()
                    * susceptible[j, k, t] * (1 + exploration_rel_tol) * estimated_infectious[j, t]
                    - self.progression_rate * exposed[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            infectious[j, k, t + 1] - infectious[j, k, t] >= (
                    self.progression_rate * exposed[j, k, t]
                    - self.detection_rate * infectious[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            hospitalized_dying[j, k, t + 1] - hospitalized_dying[j, k, t] >= (
                    self.ihd_transition_rate[k] * infectious[j, k, t]
                    - self.death_rate * hospitalized_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            quarantined_dying[j, k, t + 1] - quarantined_dying[j, k, t] >= (
                    self.iqd_transition_rate[k] * infectious[j, k, t]
                    - self.death_rate * quarantined_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            undetected_dying[j, k, t + 1] - undetected_dying[j, k, t] >= (
                    self.iud_transition_rate[k] * infectious[j, k, t]
                    - self.death_rate * undetected_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            deceased[j, k, t + 1] - deceased[j, k, t] >= self.death_rate * (
                    hospitalized_dying[j, k, t] + quarantined_dying[j, k, t] + undetected_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )

        # Set bounding constraint on absolute error of estimated infectious
        solver.addConstrs(
            infectious_error[j, t] <= max(exploration_rel_tol * estimated_infectious[j, t], exploration_abs_tol)
            for j in self._regions for t in self._timesteps
        )
        solver.addConstrs(
            infectious_error[j, t] >= estimated_infectious[j, t] - infectious.sum(j, "*", t)
            for j in self._regions for t in self._timesteps
        )
        solver.addConstrs(
            infectious_error[j, t] >= - estimated_infectious[j, t] + infectious.sum(j, "*", t)
            for j in self._regions for t in self._timesteps
        )

        # Set resource constraints
        solver.addConstrs(
            vaccinated.sum("*", "*", t) <= self.vaccine_budget[t]
            for t in self._planning_timesteps
        )
        solver.addConstrs(
            vaccinated[j, k, t] <= susceptible[j, k, t] - (1 - self.vaccine_effectiveness)
            * gp.quicksum(vaccinated[j, k, u] for u in self._planning_timesteps if u < t)
            for j in self._regions for k in self._included_risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            vaccinated[j, k, t] == 0
            for j in self._regions for k in self._included_risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            vaccinated[j, k, t] == 0
            for j in self._regions for k in self.excluded_risk_classes for t in self._timesteps
        )
        if self.optimize_capacity:
            solver.addConstrs(
                capacity[j] >= self.min_allocation_pct * self.initial_susceptible[j, self._included_risk_classes].sum()
                for j in self._regions
            )
            solver.addConstrs(
                capacity[j] <= self.max_allocation_pct * self.population[j, :].sum()
                for j in self._regions
            )
            solver.addConstrs(
                vaccinated[j, :, t].sum() <= capacity[j]
                for j in self._regions for t in self._planning_timesteps
            )
            solver.addConstrs(
                capacity.sum() <= self.max_total_capacity
            )
        else:
            solver.addConstrs(
                vaccinated.sum(j, "*", t) >= self.min_allocation_pct * (
                    susceptible.sum(j, "*", t) - (1 - self.vaccine_effectiveness)
                    * gp.quicksum(vaccinated.sum(j, "*", u) for u in self._planning_timesteps if u < t)
                )
                for j in self._regions for t in self._planning_timesteps
            )
            solver.addConstrs(
                vaccinated.sum(j, "*", t) <= self.max_allocation_pct * self.population[j, :].sum()
                for j in self._regions for t in self._planning_timesteps
            )

        # Set constraints for surplus and unallocated vaccines
        solver.addConstrs(
            surplus_vaccines[j, k] >= vaccinated.sum(j, k, "*") - self.population[j, k]
            for j in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            unallocated_vaccines[t] >= self.vaccine_budget[t] - vaccinated.sum("*", "*", t)
            for t in self._planning_timesteps
        )

        # Set objective
        solver.setObjective(
            deceased.sum("*", "*", self._n_timesteps) + hospitalized_dying.sum("*", "*", self._n_timesteps)
            + quarantined_dying.sum("*", "*", self._n_timesteps) + undetected_dying.sum("*", "*", self._n_timesteps)
            + unallocated_vaccines.sum() + surplus_vaccines.sum(),
            GRB.MINIMIZE
        )

        # Set solver params
        if mip_gap:
            solver.params.MIPGap = mip_gap
        if feasibility_tol:
            solver.params.FeasibilityTol = feasibility_tol
        if time_limit:
            solver.params.TimeLimit = time_limit
        if disable_crossover:
            solver.params.Method = 2
            solver.params.Crossover = 0
        solver.params.OutputFlag = output_flag

        # Solve model
        solver.optimize()

        # Return vaccine allocation
        vaccinated = solver.getAttr("x", vaccinated)
        vaccinated = np.array([
            [[vaccinated[j, k, t] for t in range(self._n_timesteps + 1)] for k in self._risk_classes]
            for j in self._regions
        ])
        if self.optimize_capacity:
            capacity = np.array(solver.getAttr("x", capacity))
        else:
            capacity = self.population.sum(axis=1) * self.max_allocation_pct
        return vaccinated, capacity

    def solve(
            self,
            exploration_rel_tol: float = 0.1,
            exploration_abs_tol: float = 10.0,
            termination_tol: float = 1e-2,
            mip_gap: Optional[float] = None,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            disable_crossover: bool = False,
            output_flag: bool = False,
            n_restarts: int = 10,
            max_iterations: int = 10,
            log: bool = False,
            seed: int = 0
    ) -> Tuple[DiscreteDELPHISolution, List[List[float]]]:
        """
        Solve the prescriptive DELPHI model for vaccine allocation using a coordinate descent heuristic.
        :param exploration_rel_tol: a float in [0, 1] that specifies maximum allowed relative error between the
            estimated  and actual infectious population in any region (default 0.05)
        :param exploration_abs_tol: a float  that specifies maximum allowed absolute error between the
            estimated and actual infectious population in any region (default 10.0)
        :param termination_tol: a positive float that specifies maximum allowed absolute error between the
            estimated and actual infectious population in any region
        :param mip_gap: a float that specifies the maximum MIP gap required for termination  (default 1e-2)
        :param feasibility_tol: a float that specifies that maximum feasibility tolerance for constraints (default 1e-2)
        :param time_limit: a float that specifies the maximum solve time in seconds (default 30.0)
         :param disable_crossover: a boolean that if true disables Gurobi's crossover algorithm, which used to clean up
            the interior solution of the barrier method into a basic feasible solution (default False)
        :param output_flag: a boolean that specifies whether to show the solver logs (default False)
        :param n_restarts: an integer that specifies the number of random restarts (default 10)
        :param max_iterations: an integer that specifies the maximum number of descent iterations per trial (default 10)
        :param seed: an integer that is used to set the numpy seed
        :param log:
        :return: a tuple containing a DiscreteDELPHISolution object and list of lists of float representing the descent
            trajectories
        """

        # Initialize algorithm
        np.random.seed(seed)
        trajectories = []
        best_solution = None
        best_objective = np.inf

        for restart in range(n_restarts):

            # Initialize a feasible solution
            trajectory = []
            incumbent_solution = self.simulate(randomize_allocation=True)
            incumbent_objective = incumbent_solution.get_objective()

            if log:
                print(f"Restart: {restart + 1}/{n_restarts}")
                print(f"Iteration: 0/{max_iterations} \t Objective value: {'{0:.2f}'.format(incumbent_objective)}")

            for i in range(max_iterations):

                # Store incumbent objective in trajectory
                trajectory.append(incumbent_objective)

                # Re-optimize vaccine allocation by solution linearized relaxation
                try:
                    vaccinated, capacity = self._optimize_relaxation(
                        estimated_infectious=incumbent_solution.infectious.sum(axis=1),
                        exploration_rel_tol=exploration_rel_tol,
                        exploration_abs_tol=exploration_abs_tol,
                        mip_gap=mip_gap,
                        feasibility_tol=feasibility_tol,
                        time_limit=time_limit,
                        disable_crossover=disable_crossover,
                        output_flag=output_flag
                    )
                except GurobiError:
                    if log:
                        print("Infeasible relaxation - terminating search")
                    break

                # Update incumbent solution
                previous_solution, incumbent_solution = incumbent_solution, self.simulate(vaccinated=vaccinated)
                previous_objective, incumbent_objective = incumbent_objective, incumbent_solution.get_objective()
                incumbent_solution.capacity = capacity
                if log:
                    print(f"Iteration: {i + 1}/{max_iterations} \t Objective value: {'{0:.2f}'.format(incumbent_objective)}")

                # Terminate if solution convergences
                objective_change = abs(previous_objective - incumbent_objective)
                estimated_infectious_change = np.abs(previous_solution.infectious.sum(axis=1)
                                                     - incumbent_solution.infectious.sum(axis=1)).mean()
                if max(objective_change, estimated_infectious_change) < termination_tol:
                    trajectory.append(incumbent_objective)
                    if log:
                        print("No improvement found - terminating search")
                    break

            # Store trajectory for completed trial
            trajectories.append(trajectory)

            # If the policy produced during the random trial is better than current one, update all parameters
            if incumbent_objective < best_objective:
                best_objective = incumbent_objective
                best_solution = incumbent_solution

        return best_solution, trajectories
