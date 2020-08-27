from typing import Tuple, Union, Optional, Dict

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from gurobipy import GurobiError


class DELPHISolution:

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
            eligible: np.ndarray,
            vaccinated: np.ndarray,
            capacity: Optional[np.ndarray] = None,
            days_per_timestep: float = 1.0
    ):
        """
        Instantiate a container object for a DELPHI solution.

        :param susceptible: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        susceptible population by region and risk class at each timestep
        :param exposed: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the exposed
        population by region and risk class at each timestep
        :param infectious: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        infectious population by region and risk class at each timestep
        :param hospitalized_dying: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents
        the hospitalized dying population by region and risk class at each timestep
        :param hospitalized_recovering: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that
        represents the hospitalized recovering population by region and risk class at each timestep
        :param quarantined_dying: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents
        the quarantined dying population by region and risk class at each timestep
        :param quarantined_recovering: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that
        represents the quarantined recovering population by region and risk class at each timestep
        :param undetected_dying: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        undetected dying population by region and risk class at each timestep
        :param undetected_recovering: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that
        represents the undetected recovering population by region and risk class at each timestep
        :param recovered: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        recovered population by region and risk class at each timestep
        :param deceased: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        deceased population by region and risk class at each timestep
        :param eligible: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        population eligible for vaccination by region and risk class at each timestep
        :param vaccinated: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        number vaccines allocated by region and risk class at each timestep
        :param capacity: a numpy array of shape (n_regions,) that represents the allocation capacity of each region
        per timestep
        :param days_per_timestep: a positive float that specifies the number of days per timestep used in the
        discretization scheme of the solution
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
        self.eligible = eligible
        self.vaccinated = vaccinated
        self.capacity = capacity
        self.days_per_timestep = days_per_timestep

    def get_total_deaths(self) -> float:
        return self.deceased[:, :, -1].sum()

    def get_objective_value(self) -> float:
        return (self.deceased + self.hospitalized_dying + self.quarantined_dying + self.undetected_dying)[:, :, -2].sum()

    def get_total_cases(self) -> float:
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
        Instantiate a prescriptive DELPHI model with initial conditions and parameter estimates.

        :param initial_conditions: a dictionary containing initial condition arrays
        :param delphi_params: a dictionary containing the estimated DELPHI model parameters
        :param vaccine_params: a dictionary containing vaccine-related parameters
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
        self.hospitalized_recovery_rate = delphi_params["hospitalized_recovery_rate"]
        self.unhospitalized_recovery_rate = delphi_params["unhospitalized_recovery_rate"]
        self.mortality_rate = delphi_params["mortality_rate"]
        self.days_per_timestep = delphi_params["days_per_timestep"]

        # Set vaccine parameters
        self.vaccine_effectiveness = vaccine_params["vaccine_effectiveness"]
        self.vaccine_budget = vaccine_params["vaccine_budget"]
        self.max_total_capacity = vaccine_params["max_total_capacity"]
        self.max_allocation_pct = vaccine_params["max_allocation_pct"]
        self.min_allocation_pct = vaccine_params["min_allocation_pct"]
        self.max_decrease_pct = vaccine_params["max_decrease_pct"]
        self.max_increase_pct = vaccine_params["max_increase_pct"]
        self.optimize_capacity = vaccine_params["optimize_capacity"]

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

        # Storage attributes
        self._trajectories = []
        self._solutions = []

    def simulate(
            self,
            vaccinated: Optional[np.ndarray] = None,
            randomize_allocation: bool = False,
            prioritize_allocation: bool = False
    ) -> DELPHISolution:
        """
        Solve DELPHI IVP using a forward difference scheme.

        :param vaccinated: a numpy array of (n_regions, n_classes, n_timesteps + 1) that represents a feasible
        allocation of vaccines by region and risk class at each timestep
        :param randomize_allocation: a boolean that specifies whether to randomly generate a feasible  allocation policy
        based on an exponential distribution if none provided (default False)
        :param prioritize_allocation: a boolean that specifies whether to prioritize allocation to high risk
        individuals within each region if no allocation policy is provided (default False)
        :return: a DELPHISolution object
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
        eligible = np.zeros(dims)

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
        eligible[:, :, 0] = self.initial_susceptible

        # Generate risk classes and region ranks
        risk_class_priority_rankings = np.argsort(-self.mortality_rate.mean(axis=(0, 2)))
        region_priority_rankings = np.random.permutation(self._regions)  # only used for randomized allocation

        # Propagate discrete DELPHI dynamics with vaccine allocation heuristic
        for t in self._timesteps:

            # Get eligible population subsets for vaccination
            eligible[:, self._included_risk_classes, t] = susceptible[:, self._included_risk_classes, t] \
                       - (1 - self.vaccine_effectiveness) * vaccinated[:, self._included_risk_classes, :t].sum(axis=2)
            eligible = np.maximum(eligible, 0)

            # Allocate vaccines if required
            if eligible.sum() and allocate_vaccines:

                # If random allocation specified, generate feasible allocation
                if randomize_allocation:
                    min_vaccinated = self.min_allocation_pct * eligible[:, :, t]
                    additional_budget = self.vaccine_budget[t] - min_vaccinated.sum()
                    for j in region_priority_rankings:
                        regional_budget = np.minimum(
                            additional_budget,
                            self.max_allocation_pct * self.population[j, :].sum() - min_vaccinated[j, :].sum()
                        )
                        additional_budget -= regional_budget
                        for k in risk_class_priority_rankings:
                            if k in self._included_risk_classes:
                                vaccinated[j, k, t] = np.minimum(regional_budget, eligible[j, k, t]) \
                                    + min_vaccinated[j, k]
                                regional_budget -= vaccinated[j, k, t]
                                if regional_budget <= 0:
                                    break
                        if additional_budget <= 0:
                            break

                # Else use baseline policy that orders region-wise allocation by risk class
                else:
                    if prioritize_allocation:
                        regional_budget = np.minimum(
                            eligible[:, :, t].sum(axis=1) / eligible[:, :, t].sum() * self.vaccine_budget[t],
                            self.max_allocation_pct * self.population.sum(axis=1)
                        )
                        for k in risk_class_priority_rankings:
                            if k in self._included_risk_classes:
                                vaccinated[:, k, t] = np.minimum(regional_budget, eligible[:, k, t])
                                regional_budget -= vaccinated[:, k, t]
                                if regional_budget.sum() == 0:
                                    break
                    else:
                        for k in self._included_risk_classes:
                            vaccinated[:, k, t] = self.vaccine_budget[t] * self.population[:, k] / \
                                                  self.population[:, self._included_risk_classes].sum()

            # Else ensure that the allocated vaccines do not exceed the eligible population
            vaccinated[:, :, t] = np.minimum(vaccinated[:, :, t], eligible[:, :, t])

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
                    self.ihd_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.death_rate[:, None] * hospitalized_dying[:, :, t]
            ) * self.days_per_timestep
            hospitalized_dying[:, :, t + 1] = np.maximum(hospitalized_dying[:, :, t + 1], 0)

            hospitalized_recovering[:, :, t + 1] = hospitalized_recovering[:, :, t] + (
                    self.ihr_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.hospitalized_recovery_rate * hospitalized_recovering[:, :, t]
            ) * self.days_per_timestep
            hospitalized_recovering[:, :, t + 1] = np.maximum(hospitalized_recovering[:, :, t + 1], 0)

            quarantined_dying[:, :, t + 1] = quarantined_dying[:, :, t] + (
                    self.iqd_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.death_rate[:, None] * quarantined_dying[:, :, t]
            ) * self.days_per_timestep
            quarantined_dying[:, :, t + 1] = np.maximum(quarantined_dying[:, :, t + 1], 0)

            quarantined_recovering[:, :, t + 1] = quarantined_recovering[:, :, t] + (
                    self.iqr_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.unhospitalized_recovery_rate * quarantined_recovering[:, :, t]
            ) * self.days_per_timestep
            quarantined_recovering[:, :, t + 1] = np.maximum(quarantined_recovering[:, :, t + 1], 0)

            undetected_dying[:, :, t + 1] = undetected_dying[:, :, t] + (
                    self.iud_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.death_rate[:, None] * undetected_dying[:, :, t]
            ) * self.days_per_timestep
            undetected_dying[:, :, t + 1] = np.maximum(undetected_dying[:, :, t + 1], 0)

            undetected_recovering[:, :, t + 1] = undetected_recovering[:, :, t] + (
                    self.iur_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.unhospitalized_recovery_rate * undetected_recovering[:, :, t]
            ) * self.days_per_timestep
            undetected_recovering[:, :, t + 1] = np.maximum(undetected_recovering[:, :, t + 1], 0)

            deceased[:, :, t + 1] = deceased[:, :, t] + self.death_rate[:, None] * (
                    hospitalized_dying[:, :, t] + quarantined_dying[:, :, t] + undetected_dying[:, :, t]
            ) * self.days_per_timestep

            recovered[:, :, t + 1] = recovered[:, :, t] + (
                    self.hospitalized_recovery_rate * hospitalized_recovering[:, :, t]
                    + self.unhospitalized_recovery_rate * (quarantined_recovering[:, :, t] + undetected_recovering[:, :, t])
            ) * self.days_per_timestep

        return DELPHISolution(
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
            eligible=eligible,
            vaccinated=vaccinated,
            days_per_timestep=self.days_per_timestep,
        )

    def _optimize_relaxation(
            self,
            exploration_tol: float,
            estimated_infectious: np.ndarray,
            vaccinated_warm_start: Optional[np.ndarray],
            mip_gap: Optional[float],
            barrier_conv_tol: Optional[float],
            feasibility_tol: Optional[float],
            time_limit: Optional[float],
            disable_crossover: bool,
            output_flag: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a linear relaxation of the vaccine allocation problem, based on estimated infectious populations.

        :param exploration_tol: a positive float that specifies maximum allowed absolute error between the
        estimated and actual infectious population in any region
        :param estimated_infectious: numpy array of size (n_regions, k_classes, t_timesteps + 1) that represents the
        estimated infectious population by region and risk class at each timestep, based on a previous feasible
        solution
        :param vaccinated_warm_start: an optional numpy array of size (n_regions, k_classes, t_timesteps + 1) that
        represents the initial guess for the optimal vaccine allocation policy
        :param mip_gap: an optional float that if set overrides Gurobi's default maximum MIP gap required for
        termination
        :param barrier_conv_tol: an optional float that if set overrides Gurobi's default convergence tolerance for the
        barrier method
        :param feasibility_tol: an optional float that if set overrides Gurobi's default maximum feasibility tolerance
        for constraints
        :param time_limit: an optional float that if set specifies the maximum solve time in seconds
        :param disable_crossover: a  boolean that if true disables Gurobi's crossover algorithm, which used to clean up
        the interior solution of the barrier method into a basic feasible solution
        :param output_flag: a boolean that specifies whether to show the solver logs
        :return: a tuple of two numpy arrays of size (n_regions, n_risk_classes, t_timesteps + 1) and (n_regions,) that
        respectively represent the vaccine allocation policy and the regional allocation capacities
        allocations
        """

        # Initialize model
        model = gp.Model()

        # Define decision variables
        susceptible = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        exposed = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        infectious = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        hospitalized_dying = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        quarantined_dying = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        undetected_dying = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        deceased = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        vaccinated = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        eligible = model.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        infectious_error = model.addVars(self._n_regions, self._n_timesteps, lb=0)
        surplus_vaccines = model.addVars(self._n_regions, self._n_risk_classes, lb=0)
        unallocated_vaccines = model.addVars(self._n_timesteps, lb=0)
        if self.optimize_capacity:
            capacity = model.addVars(self._n_regions, lb=0)

        # Set initial conditions for DELPHI model
        model.addConstrs(
            susceptible[j, k, 0] == self.initial_susceptible[j, k]
            for j in self._regions for k in self._risk_classes
        )
        model.addConstrs(
            exposed[j, k, 0] == self.initial_exposed[j, k]
            for j in self._regions for k in self._risk_classes
        )
        model.addConstrs(
            infectious[j, k, 0] == self.initial_infectious[j, k]
            for j in self._regions for k in self._risk_classes
        )
        model.addConstrs(
            hospitalized_dying[j, k, 0] == self.initial_hospitalized_dying[j, k]
            for j in self._regions for k in self._risk_classes
        )
        model.addConstrs(
            quarantined_dying[j, k, 0] == self.initial_quarantined_dying[j, k]
            for j in self._regions for k in self._risk_classes
        )
        model.addConstrs(
            undetected_dying[j, k, 0] == self.initial_undetected_dying[j, k]
            for j in self._regions for k in self._risk_classes
        )
        model.addConstrs(
            deceased[j, k, 0] == 0
            for j in self._regions for k in self._risk_classes
        )

        # Set terminal conditions
        model.addConstrs(
            vaccinated[j, k, self._n_timesteps] == 0
            for j in self._regions for k in self._risk_classes
        )

        # Set DELPHI dynamics constraints
        model.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] + self.vaccine_effectiveness * vaccinated[j, k, t] >=
            - self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum()
            * (susceptible[j, k, t] - self.vaccine_effectiveness * vaccinated[j, k, t])
            * estimated_infectious[j, t] * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        model.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    self.infection_rate[j] * self.policy_response[j, t] / self.population[j, :].sum()
                    * (susceptible[j, k, t] - self.vaccine_effectiveness * vaccinated[j, k, t])
                    * estimated_infectious[j, t]
                    - self.progression_rate * exposed[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        model.addConstrs(
            infectious[j, k, t + 1] - infectious[j, k, t] >= (
                    self.progression_rate * exposed[j, k, t]
                    - self.detection_rate * infectious[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        model.addConstrs(
            hospitalized_dying[j, k, t + 1] - hospitalized_dying[j, k, t] >= (
                    self.ihd_transition_rate[j, k, t] * infectious[j, k, t]
                    - self.death_rate[j] * hospitalized_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        model.addConstrs(
            quarantined_dying[j, k, t + 1] - quarantined_dying[j, k, t] >= (
                    self.iqd_transition_rate[j, k, t] * infectious[j, k, t]
                    - self.death_rate[j] * quarantined_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        model.addConstrs(
            undetected_dying[j, k, t + 1] - undetected_dying[j, k, t] >= (
                    self.iud_transition_rate[j, k, t] * infectious[j, k, t]
                    - self.death_rate[j] * undetected_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )
        model.addConstrs(
            deceased[j, k, t + 1] - deceased[j, k, t] >= self.death_rate[j] * (
                     hospitalized_dying[j, k, t] + quarantined_dying[j, k, t] + undetected_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._timesteps
        )

        # Set bounding constraint on absolute error of estimated infectious
        model.addConstrs(
            infectious_error[j, t] <= exploration_tol
            for j in self._regions for t in self._timesteps
        )
        model.addConstrs(
            infectious_error[j, t] >= estimated_infectious[j, t] - infectious.sum(j, "*", t)
            for j in self._regions for t in self._timesteps
        )
        model.addConstrs(
            infectious_error[j, t] >= infectious.sum(j, "*", t) - estimated_infectious[j, t]
            for j in self._regions for t in self._timesteps
        )

        # Set resource constraints
        model.addConstrs(
            eligible[j, k, t] == susceptible[j, k, t] - (1 - self.vaccine_effectiveness)
            * gp.quicksum(vaccinated[j, k, u] for u in self._timesteps if u < t)
            for j in self._regions for k in self._included_risk_classes for t in self._timesteps
        )
        model.addConstrs(
            vaccinated.sum("*", "*", t) <= self.vaccine_budget[t]
            for t in self._timesteps
        )
        model.addConstrs(
            vaccinated[j, k, t] <= eligible[j, k, t]
            for j in self._regions for k in self._included_risk_classes for t in self._timesteps
        )
        model.addConstrs(
            vaccinated[j, k, t] == 0
            for j in self._regions for k in self.excluded_risk_classes for t in self._timesteps
        )
        model.addConstrs(
            vaccinated.sum(j, "*", t) >= self.min_allocation_pct * eligible.sum(j, "*", t)
            for j in self._regions for t in self._timesteps
        )
        model.addConstrs(
            vaccinated.sum(j, "*", t + 1) >= vaccinated.sum(j, "*", t)
            - self.max_allocation_pct * self.population[j, :].sum() * self.max_increase_pct
            for j in self._regions for t in self._timesteps[:-1]
        )
        model.addConstrs(
            vaccinated.sum(j, "*", t + 1) <= vaccinated.sum(j, "*", t)
            + self.max_allocation_pct * self.population[j, :].sum() * self.max_decrease_pct
            for j in self._regions for t in self._timesteps[:-1]
        )
        if self.optimize_capacity:
            model.addConstrs(
                capacity[j] >= self.min_allocation_pct * self.initial_susceptible[j, self._included_risk_classes].sum()
                for j in self._regions
            )
            model.addConstrs(
                capacity[j] <= self.max_allocation_pct * self.population[j, :].sum()
                for j in self._regions
            )
            model.addConstrs(
                vaccinated[j, :, t].sum() <= capacity[j]
                for j in self._regions for t in self._timesteps
            )
            model.addConstrs(
                capacity.sum() <= self.max_total_capacity
            )
        else:
            model.addConstrs(
                vaccinated.sum(j, "*", t) <= self.max_allocation_pct * self.population[j, :].sum()
                for j in self._regions for t in self._timesteps
            )

        # Set constraints for surplus and unallocated vaccines
        model.addConstrs(
            surplus_vaccines[j, k] >= vaccinated.sum(j, k, "*") - self.population[j, k]
            for j in self._regions for k in self._risk_classes
        )
        model.addConstrs(
            unallocated_vaccines[t] >= self.vaccine_budget[t] - vaccinated.sum("*", "*", t)
            for t in self._timesteps
        )

        # Set objective
        model.setObjective(
            deceased.sum("*", "*", self._n_timesteps) + hospitalized_dying.sum("*", "*", self._n_timesteps)
            + quarantined_dying.sum("*", "*", self._n_timesteps) + undetected_dying.sum("*", "*", self._n_timesteps)
            + unallocated_vaccines.sum() + surplus_vaccines.sum(),
            GRB.MINIMIZE
        )

        # Set warm start
        if vaccinated_warm_start is not None:
            for j in self._regions:
                for k in self._included_risk_classes:
                    for t in self._timesteps:
                        vaccinated[j, k, t].start = vaccinated_warm_start[j, k, t]

        # Set solver params
        if mip_gap:
            model.params.MIPGap = mip_gap
        if barrier_conv_tol:
            model.params.BarConvTol = barrier_conv_tol
        if feasibility_tol:
            model.params.FeasibilityTol = feasibility_tol
        if time_limit:
            model.params.TimeLimit = time_limit
        if disable_crossover:
            model.params.Method = 2
            model.params.Crossover = 0
        model.params.OutputFlag = output_flag

        # Solve model
        model.optimize()

        # Return vaccine allocation
        vaccinated = model.getAttr("x", vaccinated)
        vaccinated = np.array([
            [[vaccinated[j, k, t] for t in range(self._n_timesteps + 1)] for k in self._risk_classes]
            for j in self._regions
        ])
        if self.optimize_capacity:
            capacity = np.array(model.getAttr("x", capacity))
        else:
            capacity = self.population.sum(axis=1) * self.max_allocation_pct
        return vaccinated, capacity

    def _smooth_vaccine_allocation(
            self,
            solution: DELPHISolution,
            smoothing_window: int
    ) -> DELPHISolution:
        """
        Apply a rolling-window smoothing heuristic to the vaccine allocation policy as a post-processing step.

        :param solution: a DELPHISolution object
        :param smoothing_window: an integer that specifies the symmetric smoothing window size (default)
        :return: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1) representing the smoothed vaccine
        allocation policy
        """
        vaccinated = np.zeros(solution.vaccinated.shape)
        for t in self._timesteps:
            start = max(t - smoothing_window, 0)
            end = min(t + smoothing_window, self._n_timesteps) + 1
            vaccinated[:, :, t] = solution.vaccinated[:, :, start:end].mean(axis=2)
            vaccinated[:, :, t] = vaccinated[:, :, t] * self.vaccine_budget[t] / vaccinated[:, :, t].sum()
        return self.simulate(vaccinated=vaccinated)

    def _prioritize_vaccine_allocation(self, solution: DELPHISolution) -> DELPHISolution:
        """
        Adjust vaccine allocation to ensure risk class prioritization is preserved.

        :param solution: a DELPHISolution object
        :return: a DELPHISolution object
        """

        # Rank risk classes by mortality rate
        priority_ranking = np.argsort(-self.mortality_rate.mean(axis=(0, 2)))

        for t in self._timesteps:

            # Initialize variables for timestep
            eligible = solution.eligible
            vaccinated = solution.vaccinated
            update = False

            # Perform vaccine transfers to prioritize by risk class
            for i, k in enumerate(priority_ranking):
                for l in priority_ranking[i + 1:]:
                    if k in self._included_risk_classes and l in self._included_risk_classes:
                        transfer = np.minimum(vaccinated[:, l, t], eligible[:, k, t] - vaccinated[:, k, t])
                        if transfer.max() > 0:
                            vaccinated[:, k, t] = vaccinated[:, k, t] + transfer
                            vaccinated[:, l, t] = vaccinated[:, l, t] - transfer
                            update = True

            # If a transfer was
            if update:
                solution = self.simulate(vaccinated=vaccinated)

        return solution

    def _round_vaccine_allocation(self,  solution: DELPHISolution, rounding_tol: float):
        """
        Apply a rounding cut-off to the vaccine allocation policy as a proxy for transforming the interior point
        solution into a basic feasible solution.

        :param solution: a DELPHISolution object
        :param rounding_tol: a float the specifies the maximum allocation amount that will be rounded to 0 in
        post-processing (default 1e-3)
        :return: a DELPHISolution object
        """
        vaccinated = np.where(solution.vaccinated > rounding_tol, solution.vaccinated, 0)
        for t in self._timesteps:
            vaccinated[:, :, t] = vaccinated[:, :, t] * self.vaccine_budget[t] / vaccinated[:, :, t].sum()
        return self.simulate(vaccinated=vaccinated)

    def _post_process_solution(
            self,
            solution: DELPHISolution,
            prioritize_allocation: bool,
            smooth_allocation: bool,
            round_allocation: bool,
            smoothing_window: int,
            rounding_tol: float
    ):
        """

        :param solution: a DELPHISolution object
        :param smoothing_window: an integer that specifies the symmetric smoothing window size (default)
        :param rounding_tol: a float the specifies the maximum allocation amount that will be rounded to 0 in
        post-processing (default 1e-3)
        :return: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1) representing the smoothed vaccine
        allocation policy
        """
        if smooth_allocation:
            solution = self._smooth_vaccine_allocation(solution=solution, smoothing_window=smoothing_window)
        if prioritize_allocation:
            solution = self._prioritize_vaccine_allocation(solution=solution)
        if round_allocation:
            solution = self._round_vaccine_allocation(solution=solution, rounding_tol=rounding_tol)
        return solution

    def optimize(
            self,
            exploration_tol: float,
            termination_tol: float = 1e-2,
            mip_gap: Optional[float] = None,
            barrier_conv_tol: Optional[float] = None,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            disable_crossover: bool = True,
            output_flag: bool = False,
            n_restarts: int = 1,
            max_iterations: int = 10,
            n_early_stopping_iterations: int = 2,
            smooth_allocation: bool = False,
            prioritize_allocation: bool = False,
            smoothing_window: int = 1,
            rounding_tol: float = 1e-2,
            log: bool = False,
            seed: int = 0
    ) -> DELPHISolution:
        """
        Solve the prescriptive DELPHI model for vaccine allocation using a coordinate descent heuristic.

        :param exploration_tol: a float  that specifies maximum allowed absolute error between the
        estimated and actual infectious population in any region
        :param termination_tol: a positive float that specifies maximum allowed absolute error between the
        estimated and actual infectious population in any region (default 1e-3)
        :param mip_gap: an optional float that if set overrides Gurobi's default maximum MIP gap required for
        termination (default None)
        :param barrier_conv_tol: an optional float that if set overrides Gurobi's default convergence tolerance for the
        barrier method (default None)
        :param feasibility_tol: an optional float that if set overrides Gurobi's default maximum feasibility tolerance
        for constraints (default None)
        :param time_limit: an optional float that if set specifies the maximum solve time in seconds (default None)
        :param disable_crossover: a  boolean that if true disables Gurobi's crossover algorithm, which used to clean up
        the interior solution of the barrier method into a basic feasible solution (default False)
        :param output_flag: a boolean that specifies whether to show the solver logs (default False)
        :param n_restarts: an integer that specifies the number of restarts, with a smart start provided if set to 1
        else randomized starts provided (default 1)
        :param max_iterations: an integer that specifies the maximum number of descent iterations per trial (default 10)
        :param n_early_stopping_iterations: an integer that specifies the number of descent iterations after which a
        trial is terminated if there is no improvement (default 2)
        :param smooth_allocation: a boolean that if true applies an additional smoothing heuristic to the solution in
        post-processing(default True)
        :param prioritize_allocation: a boolean
        :param smoothing_window: an integer that specifies the symmetric smoothing window size (default)
        :param rounding_tol: a float the specifies the maximum allocation amount that will be rounded to 0 in
        post-processing (default 1e-3)
        :param log: a boolean that specifies whether to log progress
        :param seed: an integer that is used to set the numpy seed
        :return: a DELPHISolution object
        """

        # Initialize algorithm
        np.random.seed(seed)
        best_solution = None
        best_obj_val = np.inf

        for restart in range(n_restarts):

            # Initialize restart
            incumbent_solution = self.simulate(
                randomize_allocation=n_restarts > 1,
                prioritize_allocation=True
            )
            incumbent_obj_val = incumbent_solution.get_objective_value()
            trajectory = [incumbent_obj_val]
            best_solution_for_restart = None
            best_obj_val_for_restart = np.inf
            n_iters_since_improvement = 0

            if log:
                print(f"Restart: {restart + 1}/{n_restarts}")
                print(f"Iteration: 0/{max_iterations} \t Objective value: {'{0:.2f}'.format(incumbent_obj_val)}")

            for i in range(max_iterations):

                # Re-optimize vaccine allocation by solution linearized relaxation
                try:
                    vaccinated, capacity = self._optimize_relaxation(
                        exploration_tol=exploration_tol,
                        estimated_infectious=incumbent_solution.infectious.sum(axis=1),
                        vaccinated_warm_start=incumbent_solution.vaccinated,
                        mip_gap=mip_gap,
                        barrier_conv_tol=barrier_conv_tol,
                        feasibility_tol=feasibility_tol,
                        time_limit=time_limit,
                        disable_crossover=disable_crossover,
                        output_flag=output_flag
                    )
                except GurobiError:
                    if log:
                        print("Infeasible relaxation - terminating search")
                    break

                # Update incumbent and previous solution
                previous_solution, incumbent_solution = incumbent_solution, self.simulate(vaccinated=vaccinated)
                previous_obj_val, incumbent_obj_val = incumbent_obj_val, incumbent_solution.get_objective_value()
                incumbent_solution.capacity = capacity
                trajectory.append(incumbent_obj_val)
                if log:
                    print(f"Iteration: {i + 1}/{max_iterations} \t Objective value: {'{0:.2f}'.format(incumbent_obj_val)}")

                # Update best solution if incumbent solution is an improvement
                if incumbent_obj_val < best_obj_val:
                    best_obj_val = incumbent_obj_val
                    best_solution = incumbent_solution

                # Update incumbent solution for restart if incumbent solution is an improvement
                if incumbent_obj_val < best_obj_val_for_restart:
                    best_obj_val_for_restart = incumbent_obj_val
                    best_solution_for_restart = incumbent_solution
                    n_iters_since_improvement = 0
                else:
                    n_iters_since_improvement += 1

                # Terminate coordinate descent for restart if solution convergences
                if n_iters_since_improvement >= n_early_stopping_iterations:
                    if log:
                        print(
                            f"No improvement found in {n_early_stopping_iterations} iterations"
                            f" - terminating search for trial"
                        )
                    break

                # Terminate coordinate descent for restart if solution convergences
                objective_change = abs(previous_obj_val - incumbent_obj_val)
                estimated_infectious_change = np.abs(
                    previous_solution.infectious.sum(axis=1) - incumbent_solution.infectious.sum(axis=1)
                ).mean()
                if max(objective_change, estimated_infectious_change) < termination_tol:
                    trajectory.append(incumbent_obj_val)
                    if log:
                        print("Solution has converged - terminating search for trial")
                    break

            # Store trajectory and best solution for completed restart
            self._trajectories.append(trajectory)
            self._solutions.append(best_solution_for_restart)

        # Apply post-processing steps to solution
        if log:
            print("Post-processing solution")
        best_solution = self._post_process_solution(
            solution=best_solution,
            smooth_allocation=smooth_allocation,
            prioritize_allocation=prioritize_allocation,
            round_allocation=disable_crossover,
            smoothing_window=smoothing_window,
            rounding_tol=rounding_tol
        )
        if log:
            print(f"Objective value after post-processing: {'{0:.2f}'.format(best_solution.get_objective_value())}")
        return best_solution

