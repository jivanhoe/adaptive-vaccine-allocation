from typing import Tuple, Union, Optional, NoReturn, List, Dict

import gurobipy as gp
from gurobipy import GurobiError
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DiscreteDELPHISolution:

    def __init__(
            self,
            susceptible: np.ndarray,
            exposed: np.ndarray,
            infectious: np.ndarray,
            hospitalized: np.ndarray,
            quarantined: np.ndarray,
            undetected: np.ndarray,
            recovered: np.ndarray,
            deceased: np.ndarray,
            vaccinated: np.ndarray,
            population: np.ndarray,
            days_per_timestep: float,
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
        :param hospitalized: a numpy array of shape (n_regions, k_classes, t_timesteps + 1) that represents the
            hospitalized population by region and risk class at each timestep
        :param quarantined: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the quarantined
            population by region and risk class at each timestep
        :param undetected: a numpy array of shape (n_regions, k_classes, t_timesteps+1) that represents the undetected
            population by region and risk class at each timestep
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
        self.hospitalized = hospitalized
        self.quarantined = quarantined
        self.undetected = undetected
        self.recovered = recovered
        self.deceased = deceased
        self.vaccinated = vaccinated
        self.population = population
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
            if attr != "days_per_timestep" and value:
                assert value.shape == expected_dims, \
                    f"Invalid dimensions for {attr} array - expected {expected_dims}, received {value.shape}"
                assert np.all(value >= 0), f"Invalid {attr} array - all values must be non-negative"

    def get_total_deaths(self) -> float:
        return self.deceased[:, :, -1].sum()

    def get_total_infections(self) -> float:
        infectious = self.infectious.sum(axis=(0, 1))
        return 0.5 * (infectious[1:] + infectious[:-1]).sum() * self.days_per_timestep

    def plot(self, figsize: Tuple[float, float] = (15.0, 7.5)) -> plt.figure:
        """
        Plot a visualization of the solution showing the change in population composition over time and the cumulative
        casualties.
        :param figsize: A tuple o that specifies the dimension of the plot
        :return: a matplotlib figure object
        """
        # Initialize figure
        fig, ax = plt.subplots(ncols=2, figsize=figsize)

        # Define plot settings
        plot_settings = dict(alpha=0.7, linestyle="solid", marker="" if self.days_per_timestep < 2 else ".")

        # Get x-axis for plots
        days = np.arange(self.susceptible.shape[-1]) * self.days_per_timestep

        total_pop = self.population.sum()
        ax[0].plot(
            days, self.susceptible.sum(axis=(0, 1)) / total_pop * 100,
            label="Susceptible", color="tab:blue", **plot_settings
        )
        ax[0].plot(
            days[:-1],
            (self.vaccinated.sum(axis=(0, 1)).cumsum() + self.recovered.sum(axis=(0, 1))[:-1]) / total_pop * 100,
            label="Vaccinated or recovered", color="tab:green", **plot_settings
        )
        ax[0].plot(
            days, (self.exposed + self.infectious + self.hospitalized + self.quarantined + self.undetected
                   ).sum(axis=(0, 1)) / total_pop * 100,
            label="Exposed or infected", color="tab:orange", **plot_settings
        )
        ax[0].plot(
            days, self.deceased.sum(axis=(0, 1)) / total_pop * 100,
            label="Deceased", color="black", **plot_settings
        )

        ax[0].legend(fontsize=12)
        ax[0].set_xlabel("Days", fontsize=14)
        ax[0].set_ylabel("% of population", fontsize=14)
        ax[0].set_title("Population composition", fontsize=16)

        # Make plot of cumulative negative outcomes
        infectious = self.infectious.sum(axis=(0, 1))
        hospitalized = self.hospitalized.sum(axis=(0, 1))
        ax[1].plot(
            days[:-1], (0.5 * (infectious[1:] + infectious[:-1]) * self.days_per_timestep).cumsum(),
            label="Infections", color="tab:orange", **plot_settings
        )
        ax[1].plot(
            days[:-1], (0.5 * (hospitalized[1:] + hospitalized[:-1]) * self.days_per_timestep).cumsum(),
            label="Hospitalizations", color="tab:red", **plot_settings
        )
        ax[1].plot(
            days, self.deceased.sum(axis=(0, 1)),
            label="Deaths", color="black", **plot_settings
        )
        ax[1].legend(fontsize=12)
        ax[1].set_xlabel("Days", fontsize=14)
        ax[1].set_ylabel("Cumulative total", fontsize=14)
        ax[1].set_title("Casualties", fontsize=16)


class PrescriptiveDELPHIModel:

    def __init__(
            self,
            initial_conditions: Dict[str, np.ndarray],
            params: Dict[str, Union[float, np.ndarray]],
    ):
        """
        Instantiate a discrete DELPHI model with initial conditions and parameter estimates.
        :param initial_conditions: a dictionary containing initial condition arrays
        :param params: a dictionary containing the estimate model parameters
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
        self.vaccine_budget = initial_conditions["vaccine_budget"]

        # Set model parameters
        self.infection_rate = params["infection_rate"]
        self.policy_response = params["policy_response"]
        self.progression_rate = params["progression_rate"]
        self.detection_rate = params["detection_rate"]
        self.ihd_transition_rate = params["ihd_transition_rate"]
        self.ihr_transition_rate = params["ihr_transition_rate"]
        self.iqd_transition_rate = params["iqd_transition_rate"]
        self.iqr_transition_rate = params["iqr_transition_rate"]
        self.iud_transition_rate = params["iud_transition_rate"]
        self.iur_transition_rate = params["iur_transition_rate"]
        self.death_rate = params["death_rate"]
        self.recovery_rate = params["recovery_rate"]
        self.days_per_timestep = params["days_per_timestep"]

        # Initialize helper attributes
        self._n_regions = self.initial_susceptible.shape[0]
        self._n_risk_classes = self.initial_susceptible.shape[1]
        self._n_timesteps = self.vaccine_budget.shape[0]
        self._regions = range(self._n_regions)
        self._risk_classes = range(self._n_risk_classes)
        self._timesteps = range(self._n_timesteps)
        self._planning_timesteps = [t for t in self._timesteps if self.vaccine_budget[t] > 0]
        self._non_planning_timesteps = [t for t in self._timesteps if self.vaccine_budget[t] == 0]

        # Validate model inputs
        self._validate_inputs()

    def _validate_inputs(self) -> NoReturn:
        pass

    def simulate(
            self,
            vaccinated: Optional[np.ndarray] = None,
            randomize_allocation: bool = False,
            fairness_param: float = 0.0
    ) -> DiscreteDELPHISolution:
        """
        Solve DELPHI system using a forward difference scheme.
        :param vaccinated: a numpy array of (n_regions, n_classes, n_timesteps + 1) that represents a feasible
            allocation of vaccines by region and risk class at each timestep
        :param fairness_param: a float that specifies the minimum proportion of the susceptible population in each
            region that must be allocated a vaccine (default 0)
        :param randomize_allocation: a bool that specifies whether to randomize the allocation policy if none provided
            (default False)
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

            # Check if total susceptible population is non-zero
            total_susceptible = susceptible[:, :, t].sum()
            if total_susceptible > 0 and allocate_vaccines:

                # If random allocation specified, generate feasible allocation
                if randomize_allocation:
                    min_vaccinated = fairness_param * susceptible[:, :, t]
                    additional_budget = self.vaccine_budget[t] - min_vaccinated.sum()
                    additional_proportion = np.random.exponential(size=(self._n_regions, self._n_risk_classes)) ** 2
                    additional_proportion = additional_proportion / additional_proportion.sum()
                    vaccinated[:, :, t] = min_vaccinated + additional_proportion * additional_budget

                # Else use baseline policy that orders region-wise allocation by risk class
                else:
                    regional_budget = susceptible[:, :, t].sum(axis=1) / total_susceptible * self.vaccine_budget[t]
                    for k in np.argsort(-self.ihd_transition_rate):
                        vaccinated[:, k, t] = np.minimum(regional_budget, susceptible[:, k, t])
                        regional_budget -= vaccinated[:, k, t]

            vaccinated[:, :, t] = np.minimum(vaccinated[:, :, t], susceptible[:, :, t])

            # Apply Euler forward difference scheme with clipping of negative values
            for j in self._regions:
                susceptible[j, :, t + 1] = susceptible[j, :, t] - vaccinated[j, :, t] - (
                        self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
                        * (susceptible[j, :, t] - vaccinated[j, :, t]) * infectious[j, :, t].sum()
                ) * self.days_per_timestep
            susceptible[:, :, t + 1] = np.maximum(susceptible[:, :, t + 1], 0)

            exposed[:, :, t + 1] = exposed[:, :, t] + susceptible[:, :, t] - susceptible[:, :, t + 1] \
                                   - vaccinated[:, :, t] - self.progression_rate * exposed[:, :,
                                                                                   t] * self.days_per_timestep
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

        hospitalized = hospitalized_dying + hospitalized_recovering
        quarantined = quarantined_dying + quarantined_recovering
        undetected = undetected_dying + undetected_recovering

        return DiscreteDELPHISolution(
            susceptible=susceptible,
            exposed=exposed,
            infectious=infectious,
            hospitalized=hospitalized,
            quarantined=quarantined,
            undetected=undetected,
            recovered=recovered,
            deceased=deceased,
            vaccinated=vaccinated,
            population=self.population,
            days_per_timestep=self.days_per_timestep
        )

    def _optimize_relaxation(
            self,
            estimated_infectious: np.ndarray,
            exploration_tol: float,
            fairness_param: float,
            mip_gap: Optional[float],
            feasibility_tol: Optional[float],
            time_limit: Optional[float],
            output_flag: bool
    ) -> np.ndarray:
        """
        Solve the linear relaxation of the  for the optimal vaccine policy, given a fixed infectious pop
        :param estimated_infectious: numpy array of size (n_regions, k_classes, t_timesteps + 1) represents the
            estimated infectious population by region and risk class at each timestep, based on a previous feasible
            solution
        :param exploration_tol: a float in [0, 1] that specifies maximum allowed relative error between the estimated
            and actual infectious population in any region
        :param fairness_param: a float that specifies the minimum proportion of the susceptible population in each
            region that must be allocated a vaccine
        :param mip_gap: a float that specifies the maximum MIP gap required for termination
        :param feasibility_tol: a float that specifies that maximum feasibility tolerance for constraints
        :param output_flag: a boolean that specifies whether to show the solver logs
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
        vaccinated = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps, lb=0)
        infectious_error = solver.addVars(self._n_regions, self._n_timesteps + 1, lb=0)

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
        solver.addConstrs(
            vaccinated[j, k, t] == 0
            for j in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )

        # Set DELPHI dynamics constraints
        solver.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] + vaccinated[j, k, t] >=
            - (1 - exploration_tol) * self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
            * (susceptible[j, k, t] - vaccinated[j, k, t]) * estimated_infectious[j, t] * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] >=
            - (1 - exploration_tol) * self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
            * susceptible[j, k, t] * estimated_infectious[j, t] * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    (1 + exploration_tol) * self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
                    * (susceptible[j, k, t] - vaccinated[j, k, t]) * estimated_infectious[j, t]
                    - self.progression_rate * exposed[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    (1 + exploration_tol) * self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
                    * susceptible[j, k, t] * estimated_infectious[j, t]
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
            infectious_error[j, t] <= max(exploration_tol * estimated_infectious[j, t], 10)
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
            vaccinated.sum("*", "*", t) == self.vaccine_budget[t]
            for t in self._planning_timesteps
        )
        solver.addConstrs(
            vaccinated.sum(j, "*", t) >= fairness_param * susceptible.sum(j, "*", t)
            for j in self._regions for t in self._planning_timesteps
        )

        # Set objective
        solver.setObjective(deceased.sum("*", "*", self._n_timesteps), GRB.MINIMIZE)

        # Set solver params
        if mip_gap:
            solver.params.MIPGap = mip_gap
        if feasibility_tol:
            solver.params.FeasibilityTol = feasibility_tol
        if time_limit:
            solver.params.TimeLimit = time_limit
        solver.params.OutputFlag = output_flag

        # Solve model
        solver.optimize()

        # Return vaccine allocation
        vaccinated = solver.getAttr("x", vaccinated)
        return np.array([
            [[vaccinated[j, k, t] for t in range(self._n_timesteps)] for k in self._risk_classes]
            for j in self._regions
        ])

    def solve(
            self,
            exploration_tol: float = 0.05,
            termination_tol: float = 1e-2,
            fairness_param: float = 0.0,
            mip_gap: Optional[float] = None,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            output_flag: bool = False,
            n_restarts: int = 10,
            max_iterations: int = 10,
            log: bool = False,
            seed: int = 0
    ) -> Tuple[DiscreteDELPHISolution, List[List[float]]]:
        """
        Solve the prescriptive DELPHI model for vaccine allocation using a coordinate descent heuristic.
        :param exploration_tol: a float in [0, 1] that specifies maximum allowed relative error between the estimated
            and actual infectious population in any region (default 0.1)
        :param termination_tol:
        :param fairness_param: a float that specifies the minimum proportion of the susceptible population in each
            region that must be allocated a vaccine (default 0.0)
        :param mip_gap: a float that specifies the maximum MIP gap required for termination  (default 1e-2)
        :param feasibility_tol: a float that specifies that maximum feasibility tolerance for constraints (default 1e-2)
        :param time_limit: a float that specifies the maximum solve time in seconds (default 30.0)
        :param output_flag: a boolean that specifies whether to show the solver logs (default false)
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
            incumbent_solution = self.simulate(randomize_allocation=True, fairness_param=fairness_param)
            incumbent_objective = incumbent_solution.get_total_deaths()

            if log:
                logger.info(f"Restart: {restart + 1}/{n_restarts}")

            for i in range(max_iterations):

                if log:
                    logger.info(f"Iteration: {i + 1}/{max_iterations} \t Objective value: {incumbent_objective}")

                # Store incumbent objective in trajectory
                trajectory.append(incumbent_objective)

                # Re-optimize vaccine allocation by solution linearized relaxation
                vaccinated = self._optimize_relaxation(
                    estimated_infectious=incumbent_solution.infectious.sum(axis=1),
                    exploration_tol=exploration_tol,
                    fairness_param=fairness_param,
                    mip_gap=mip_gap,
                    feasibility_tol=feasibility_tol,
                    time_limit=time_limit,
                    output_flag=output_flag
                )

                # Update incumbent solution
                previous_solution, incumbent_solution = incumbent_solution, self.simulate(vaccinated=vaccinated)
                previous_objective, incumbent_objective = incumbent_objective, incumbent_solution.get_total_deaths()

                # Terminate if solution convergences
                objective_change = abs(previous_objective - incumbent_objective)
                estimated_infectious_change = np.abs(previous_solution.infectious.sum(axis=1)
                                                     - incumbent_solution.infectious.sum(axis=1)).sum()
                if max(objective_change, estimated_infectious_change) < termination_tol:
                    trajectory.append(incumbent_objective)
                    if log:
                        logger.info("No improvement found - terminating trial")
                    break

            # Store trajectory for completed trial
            trajectories.append(trajectory)

            # If the policy produced during the random trial is better than current one, update all parameters
            if incumbent_objective < best_objective:
                best_objective = incumbent_objective
                best_solution = incumbent_solution

        return best_solution, trajectories

    def _get_variable_value(self, solver: gp.Model, variable: gp.tupledict) -> np.array:
        """
        Get the optimized value of a decision variable.
        :param variable: a gurobipy tupledict object representing a decision variable in the model
        :return: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1)
        """
        variable = solver.getAttr('x', variable)
        return np.array([
            [[variable[i, k, t] for t in range(self._n_timesteps + 1)] for k in self._risk_classes]
            for i in self._regions
        ])

    def solve_benchmark(
            self,
            initial_vaccinated: Optional[np.ndarray] = None,
            fairness_param: float = 0.0,
            mip_gap: Optional[float] = None,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            output_flag: bool = False,
            check_warm_start_feasibility: bool = True,
            log: bool = False
    ):

        # Initialize model
        solver = gp.Model("nonconvex-DELPHI")

        # Define decision variables
        susceptible = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        exposed = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        infectious = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        hospitalized_dying = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        quarantined_dying = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        undetected_dying = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        deceased = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        vaccinated = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps, lb=0)

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
        solver.addConstrs(
            vaccinated[j, k, t] == 0
            for j in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )

        # Set DELPHI dynamics constraints
        solver.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] + vaccinated[j, k, t] >=
            - self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
            * (susceptible[j, k, t] - vaccinated[j, k, t]) * infectious.sum(j, "*", t) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] >=
            -self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
            * susceptible[j, k, t] * infectious.sum(j, "*", t) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
                    * (susceptible[j, k, t] - vaccinated[j, k, t]) * infectious.sum(j, "*", t)
                    - self.progression_rate * exposed[j, k, t]
            ) * self.days_per_timestep
            for j in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    self.infection_rate[j] * self.policy_response[j, t] / self.population[j]
                    * susceptible[j, k, t] * infectious.sum(j, "*", t)
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

        # Set resource constraints
        solver.addConstrs(
            vaccinated.sum("*", "*", t) == self.vaccine_budget[t]
            for t in self._planning_timesteps
        )
        solver.addConstrs(
            vaccinated.sum(j, "*", t) >= fairness_param * susceptible.sum(j, "*", t)
            for j in self._regions for t in self._planning_timesteps
        )

        # Set objective
        solver.setObjective(deceased.sum("*", "*", self._n_timesteps), GRB.MINIMIZE)

        # Set solver params
        solver.params.NonConvex = 2
        if mip_gap:
            solver.params.MIPGap = mip_gap
        if feasibility_tol:
            solver.params.FeasibilityTol = feasibility_tol
        if time_limit:
            solver.params.TimeLimit = time_limit
        solver.params.OutputFlag = output_flag

        # Check feasibility of warm start
        expected_dims = (self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        if check_warm_start_feasibility:
            assert initial_vaccinated, f"Invalid argument type for initial_vaccinated - " \
                                       f"got {type(initial_vaccinated)}, expected{type(np.array([]))}"

            assert initial_vaccinated.shape == expected_dims, \
                f"Invalid dimensions for initial_vaccinated - expected {expected_dims}, " \
                f"received {initial_vaccinated.shape}"
            assert np.all(initial_vaccinated >= 0), f"Invalid initial_vaccinated - all values must be non-negative"

            # Set the UB and LB to the initial variable
            solver.setAttr(GRB.Attr.LB, vaccinated, initial_vaccinated)
            solver.setAttr(GRB.Attr.UB, vaccinated, initial_vaccinated)
            solver.update()
            solver.optimize()

            if solver.status == GRB.Status.INFEASIBLE:
                if log:
                    logger.info(f"Model is infeasible")

                solver.computeIIS()
                constraint_inf = solver.IISCONSTR
                if log:
                    logger.info(f"Found {sum(constraint_inf)} out of {len(constraint_inf)} violated constraints")
                    logger.info(f"Solving model {solver.getAttr('ModelName')}, without user-provided warm start")
                    solver.setAttr(GRB.Attr.LB, vaccinated, 0)
                    solver.setAttr(GRB.Attr.UB, vaccinated, GRB.INFINITY)
                    solver.update()
                    solver.optimize()
                    best_bound = solver.ObjBound()
                    vaccines = self._get_variable_value(solver=solver, variable=vaccinated)
                    try:
                        best_objective = solver.ObjVal()
                    except GurobiError:
                        best_objective = None

                    return vaccines, best_objective, best_bound

            elif solver.status == GRB.Status.OPTIMAL:
                if log:
                    logger.info(f"Warm start is feasible \t Incumbent objective value: {solver.objVal}")
                    logger.info(f"Solving model {solver.getAttr('ModelName')}, using provided warm start")

                solver.setAttr(GRB.Attr.LB, vaccinated, 0)
                solver.setAttr(GRB.Attr.UB, vaccinated, GRB.INFINITY)
                for i in self._regions:
                    for k in self._risk_classes:
                        for t in self._timesteps:
                            vaccinated[i, k, t].start = initial_vaccinated[i, k, t]

                solver.update()
                solver.optimize()
                vaccines = self._get_variable_value(solver=solver, variable=vaccinated)
                best_objective, best_bound = solver.ObjVal, solver.ObjBound

                return vaccines, best_objective, best_bound

        # Solve model
        solver.optimize()

        # Return vaccine allocation
        best_bound = solver.ObjBound()
        vaccines = self._get_variable_value(solver=solver, variable=vaccinated)
        try:
            best_objective = solver.ObjVal()
        except GurobiError:
            best_objective = None

        return vaccines, best_objective, best_bound
