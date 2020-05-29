import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Tuple, Union, Optional, NoReturn, Any
import matplotlib.pyplot as plt


class DiscreteDELPHISolution:

    def __init__(
            self,
            susceptible: Optional[np.ndarray] = None,
            exposed: Optional[np.ndarray] = None,
            infected: Optional[np.ndarray] = None,
            hospitalized: Optional[np.ndarray] = None,
            recovered: Optional[np.ndarray] = None,
            deceased: Optional[np.ndarray] = None,
            vaccinated: Optional[np.ndarray] = None,
            days_per_timestep: Optional[float] = None,
            validate_on_init: bool = False
    ):
        """
        Instantiate a container object for a DELPHI solution
        :param susceptible: if provided, a numpy array of shape [n_regions, k_classes, t_timesteps+1].
               represents the susceptible population of a region for a given risk class at a specific timestep
        :param exposed: if provided, a numpy array of shape [n_regions, k_classes, t_timesteps+1].
               represents the exposed population of a region for a given risk class at a specific timestep
        :param infected: if provided, a numpy array of shape [n_regions, k_classes, t_timesteps+1].
               represents the infected population of a region for a given risk class at a specific timestep
        :param hospitalized: if provided, a numpy array of shape [n_regions, k_classes, t_timesteps+1].
               represents the hospitalized population of a region for a given risk class at a specific timestep
        :param recovered: if provided, a numpy array of shape [n_regions, k_classes, t_timesteps+1].
               represents the recovered population of a region for a given risk class at a specific timestep
        :param deceased: if provided, a numpy array of shape [n_regions, k_classes, t_timesteps+1].
               represents the deceased population of a region for a given class at a specific timestep
        :param vaccinated: if provided, a numpy array of shape [n_regions, k_classes, t_timesteps+1].
               represents the vaccinated population of a region for a given class at a specific timestep
        :param validate_on_init: boolean value, indicates whether to check the input dimensions are correct
        """

        # Initialize functional states
        self._susceptible = susceptible if susceptible else np.zeros(susceptible.shape)
        self._exposed = exposed if exposed else np.zeros(susceptible.shape)
        self._infected = infected if infected else np.zeros(susceptible.shape)
        self._hospitalized = hospitalized if hospitalized else np.zeros(susceptible.shape)
        self._recovered = recovered if recovered else np.zeros(susceptible.shape)
        self._deceased = deceased if deceased else np.zeros(susceptible.shape)
        self._vaccinated = vaccinated if vaccinated else np.zeros(susceptible.shape)
        self._days_per_timestep = days_per_timestep

        # Check solution
        if validate_on_init:
            self._validate_solution()

    def _validate_solution(self) -> NoReturn:
        """
        Check that the provided solution arrays have valid dimensions and values.
        :return: NoReturn
        """
        expected_dims = self._susceptible.shape
        for attr, value in self.__dict__.items():
            if attr != '_days_per_timestep' and value:
                assert value.shape == expected_dims, \
                f"Invalid dimensions for {attr} array - expected {expected_dims}, received {value.shape}"
                assert np.all(value >= 0), f"Invalid {attr} array - all values must be non-negative"

    # Define getter decorator for functional states
    @property
    def functional_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._infected, self._deceased, self._vaccinated

    # Define setter decorator for functional states
    @functional_states.setter
    def functional_states(
            self, values: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
    ) -> NoReturn:

        # Unpack tuple
        new_susceptible, new_exposed, new_infected, new_hospitalized, \
            new_recovered, new_deceased, new_vaccinated, days_per_timestep = values

        # Make assignment
        self._susceptible = new_susceptible
        self._exposed = new_exposed
        self._infected = new_infected
        self._hospitalized = new_hospitalized
        self._recovered = new_recovered
        self._deceased = new_deceased
        self._vaccinated = new_vaccinated
        self._days_per_timestep = days_per_timestep

    def plot(self, figsize: Tuple[float, float] = (15.0, 7.5)) -> plt.figure:
        """
        Plot a visualization of the solution showing the change in population composition over time and the cumulative
        number of infections and deaths.
        :param figsize: A tuple o that specifies the dimension of the plot
        :return: a matplotlib figure object
        """
        pass


class HeuristicDELPHIModel:

    def __init__(
            self,
            init_variables: dict,
            params: dict,
    ):
        """
        Instantiate a discrete DELPHI model with initial conditions and parameter estimates
        :param init_variables: dictionary object containing key-value pairs for each decision variables
        :param params: a dictionary object containing key-value pairs for each estimate parameter
        """

        # Set initial conditions
        self.initial_susceptible = init_variables.get('initial_susceptible')
        self.initial_exposed = init_variables.get('initial_exposed')
        self.initial_infected = init_variables.get('initial_infected')
        self.initial_hospitalized_dth = init_variables.get('initial_hospitalized_dth')
        self.initial_hospitalized_rcv = init_variables.get('initial_hospitalized_rcv')
        self.initial_quarantine_dth = init_variables.get('initial_quarantine_dth')
        self.initial_quarantine_rcv = init_variables.get('initial_quarantine_rcv')
        self.initial_undetected_dth = init_variables.get('initial_undetected_dth')
        self.initial_undetected_rcv = init_variables.get('initial_undetected_rcv')
        self.initial_recovered = init_variables.get('initial_recovered')
        self.population = init_variables.get('population')
        self.vaccine_budget = init_variables.get('vaccine_budget')
        self.other_vaccines = init_variables.get('other_vaccinated')

        # Set model parameters
        self.infection_rate = params.get('infection_rate')
        self.policy_response = params.get('policy_response')
        self.progression_rate = params.get('progression_rate')
        self.ihd_rate = params.get('ihd_rate')
        self.ihr_rate = params.get('ihr_rate')
        self.iqd_rate = params.get('iqd_rate')
        self.iqr_rate = params.get('iqr_rate')
        self.iud_rate = params.get('iud_rate')
        self.iur_rate = params.get('iur_rate')
        self.ith_rate = params.get('ith_rate')
        self.detection_rate = params.get('detection_rate')
        self.death_rate = params.get('death_rate')
        self.recovery_rate = params.get('recovery_rate')
        self.days_per_timestep = params.get('days_per_timestep')

        # Initialize helper attributes
        self._n_regions = self.initial_susceptible.shape[0]
        self._n_risk_classes = self.initial_susceptible.shape[1]
        self._n_timesteps = self.vaccine_budget.shape[0]
        self._regions = range(self._n_regions)
        self._risk_classes = range(self._n_risk_classes)
        self._timesteps = range(self._n_timesteps)
        self._planning_timesteps = [t for t in self._timesteps if self.vaccine_budget[t] > 0]
        self._non_planning_timesteps = [t for t in self._timesteps if self.vaccine_budget[t] == 0]

        self._validate_inputs()

    def _validate_inputs(self):
        pass

    def generate_feasible_policy(self) -> np.ndarray:
        """
        Solve heuristic model to find a random feasible vaccine policy
        :return: numpy array of dims [n_regions, k_classes, t_timesteps +1] of a feasible vaccine policy
        """
        # TODO: an idea. Start by allocating vaccines for all timesteps always in the same wqy. Then choose k-out-of-n
        # TODO: states at random, and swap some vaccines between pairs of states while ensuring that the swapping does
        # TODO: not give a total # vaccines per state > susceptible pop.
        pass

    def solve_stage1(self, vaccinated: np.ndarray) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Optional[Any]]:
        """
        Solves DELPHI model using a finite difference forward scheme. Requires feasible vaccine allocation policy.
        :param vaccinated: array of dims (n_regions, n_classes, n_timesteps +1), is a feasible vaccine allocation
        :return: Tuple object of the model's functional states
        """

        # Initialize functional states
        num_states = 13
        dims = (self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        susceptible, exposed, infected, hospitalized_dth, hospitalized_rcv, \
            quarantine_dth, quarantine_rcv, undetected_dth, undetected_rcv, \
            total_hospitalized, deceased, recovered = (np.zeros(dims) for _ in range(num_states))

        # Set initial conditions
        susceptible[:, :, 0] = self.initial_susceptible
        exposed[:, :, 0] = self.initial_exposed
        infected[:, :, 0] = self.initial_infected
        hospitalized_dth[:, :, 0] = self.initial_hospitalized_dth
        hospitalized_rcv[:, :, 0] = self.initial_hospitalized_rcv
        quarantine_dth[:, :, 0] = self.initial_quarantine_dth
        quarantine_rcv[:, :, 0] = self.initial_quarantine_rcv
        undetected_dth[:, :, 0] = self.initial_undetected_dth
        undetected_rcv[:, :, 0] = self.initial_undetected_rcv
        total_hospitalized[:, :, 0] = self.initial_hospitalized_dth + self.initial_hospitalized_rcv
        recovered[:, :, 0] = self.initial_recovered

        # Propagate discrete DELPHI dynamics with vaccine allocation heuristic
        for t in self._timesteps:

            # Check if total susceptible population is non-zero
            total_susceptible = susceptible[:, :, t].sum()
            if total_susceptible > 0:

                # Euler forward difference scheme
                for i in self._regions:
                    susceptible[i, :, t + 1] = susceptible[i, :, t] - vaccinated[i, :, t] - (
                            self.infection_rate[i]/self.population[i] * self.policy_response[i, t] *
                            (susceptible[i, :, t] - vaccinated[i, :, t]) *
                            infected[i, :, t].sum() * self.days_per_timestep
                    )
                    exposed[i, :, t + 1] = exposed[i, :, t] + (
                            self.infection_rate[i]/self.population[i] * self.policy_response[i, t] *
                            (susceptible[i, :, t] - vaccinated[i, :, t]) *
                            infected[i, :, t].sum() - self.progression_rate *
                            exposed[i, :, t]
                    ) * self.days_per_timestep

                    infected[i, :, t + 1] = infected[i, :, t] + (
                            self.progression_rate * exposed[i, :, t] - self.detection_rate * infected[i, :, t]
                    ) * self.days_per_timestep

                    hospitalized_dth[i, :, t + 1] = hospitalized_dth[i, :, t] + (
                            self.ihd_rate * infected[i, :, t] - self.death_rate * hospitalized_dth[i, :, t]
                    ) * self.days_per_timestep

                    hospitalized_rcv[i, :, t + 1] = hospitalized_rcv[i, :, t] + (
                            self.ihr_rate * infected[i, :, t] - self.recovery_rate * hospitalized_rcv[i, :, t]
                    ) * self.days_per_timestep

                    quarantine_dth[i, :, t + 1] = quarantine_dth[i, :, t] + (
                            self.iqd_rate * infected[i, :, t] - self.death_rate * quarantine_dth[i, :, t]
                    ) * self.days_per_timestep

                    quarantine_rcv[i, :, t + 1] = quarantine_rcv[i, :, t] + (
                            self.iqr_rate * infected[i, :, t] - self.recovery_rate * quarantine_rcv[i, :, t]
                    ) * self.days_per_timestep

                    undetected_dth[i, :, t + 1] = undetected_dth[i, :, t] + (
                            self.iud_rate * infected[i, :, t] - self.death_rate * undetected_dth[i, :, t]
                    ) * self.days_per_timestep

                    undetected_rcv[i, :, t + 1] = undetected_rcv[i, :, t] + (
                            self.iur_rate * infected[i, :, t] - self.recovery_rate * undetected_rcv[i, :, t]
                    ) * self.days_per_timestep

                    total_hospitalized[i, :, t+1] = self.ith_rate * infected[i, :, t] * self.days_per_timestep

                    deceased[i, :, t + 1] = deceased[i, :, t] + self.death_rate * (
                            hospitalized_dth[i, :, t] + quarantine_dth[i, :, t] + undetected_dth
                    ) * self.days_per_timestep

                    recovered[i, :, t + 1] = recovered[i, :, t] + self.recovery_rate * (
                            hospitalized_rcv[i, :, t] + quarantine_rcv[i, :, t] + undetected_rcv
                    ) * self.days_per_timestep

        return susceptible, exposed, infected, total_hospitalized, recovered, \
            deceased, vaccinated, self.days_per_timestep

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

    def solve_stage2(
            self,
            infected_estimate: np.ndarray,
            exploration_tol: float = 0.1,
            fairness_param: float = 0.0,
            mip_gap: float = 1e-2,
            feasibility_tol: float = 1e-2,
            output_flag: bool = False,
            time_limit: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the LOP for the optimal vaccine policy, given a fixed infected pop
        :param infected_estimate: numpy array of shape [n_regions, k_classes, t_timesteps +1] represents the estimated
            infected population
        :param exploration_tol: a float in [0, 1] that specifies our confidence in the infected_estimate. 0 is high.
        :param fairness_param: a float that specifies the minimum proportion of the susceptible population in each
            region that must be allocated a vaccine (default 0)
        :param mip_gap: a float that specifies the maximum MIP gap required for termination (default 1e-2)
        :param feasibility_tol: a float that specifies that maximum feasibility tolerance for constraints (default 1e-2)
        :param output_flag: a boolean that specifies whether to show the solver logs (default False)
        :param time_limit: a float that specifies the maximum solve time in seconds (default 120.0)
        :return: tuple of numpy arrays, representing the deceased and optimal vaccine decision variables
        """

        # Initialize model
        solver = gp.Model('delphi_stage2')

        # Define decision variables
        susceptible = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        exposed = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        infected = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        hospitalized_dth = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        hospitalized_rcv = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        quarantine_dth = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        quarantine_rcv = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        undetected_dth = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        undetected_rcv = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        total_hospitalized = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        recovered = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        deceased = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        vaccinated = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)

        # Define analysis variable for absolute value constraint
        abs_box_infection = solver.addVars(self._n_regions, self._n_timesteps+1)

        # Set initial conditions for DELPHI model
        solver.addConstrs(
            susceptible[i, k, 0] == self.initial_susceptible[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            exposed[i, k, 0] == self.initial_exposed[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            infected[i, k, 0] == self.initial_infected[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            hospitalized_dth[i, k, 0] == self.initial_hospitalized_dth[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            hospitalized_rcv[i, k, 0] == self.initial_hospitalized_rcv[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            quarantine_dth[i, k, 0] == self.initial_quarantine_dth[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            quarantine_rcv[i, k, 0] == self.initial_quarantine_rcv[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            undetected_dth[i, k, 0] == self.initial_undetected_dth[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            undetected_rcv[i, k, 0] == self.initial_undetected_rcv[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            total_hospitalized[i, k, 0] == self.initial_hospitalized_dth[i, k] + self.initial_hospitalized_rcv[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            recovered[i, k, 0] == self.initial_recovered[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            vaccinated[i, k, t] == 0
            for i in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            deceased[:, :, 0] == 0
        )

        # Set DELPHI dynamics constraints
        solver.addConstrs(
            susceptible[i, k, t+1] - susceptible[i, k, t] - vaccinated[i, k, t] >=
            - (1-exploration_tol) * self.infection_rate[i]/self.population[i] * self.policy_response[i, t] *
            (susceptible[i, k, t] - vaccinated[i, k, t]) * infected_estimate[i, t] * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            exposed[i, k, t+1] - exposed[i, k, t] >= (
                (1 + exploration_tol) * self.infection_rate[i]/self.population[i] * self.policy_response[i, t] *
                (susceptible[i, k, t] - vaccinated[i, k, t]) * infected_estimate[i, t] -
                self.progression_rate * exposed[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            infected[i, k, t+1] - infected[i, k, t] >= (
                self.progression_rate * exposed[i, k, t] - self.detection_rate * infected[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            hospitalized_dth[i, k, t+1] - hospitalized_dth[i, k, t] >= (
                self.ihd_rate[k] * infected[i, k, t] - self.death_rate[k] * hospitalized_dth[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            hospitalized_rcv[i, k, t + 1] - hospitalized_rcv[i, k, t] >= (
                    self.ihr_rate[k] * infected[i, k, t] - self.recovery_rate[k] * hospitalized_rcv[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            quarantine_dth[i, k, t + 1] - quarantine_dth[i, k, t] >= (
                    self.iqd_rate[k] * infected[i, k, t] - self.death_rate[k] * quarantine_dth[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            quarantine_rcv[i, k, t + 1] - quarantine_rcv[i, k, t] >= (
                    self.iqr_rate[k] * infected[i, k, t] - self.recovery_rate[k] * quarantine_rcv[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            undetected_dth[i, k, t + 1] - undetected_dth[i, k, t] >= (
                    self.iud_rate[k] * infected[i, k, t] - self.death_rate[k] * undetected_dth[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            undetected_rcv[i, k, t + 1] - undetected_rcv[i, k, t] >= (
                    self.iur_rate[k] * infected[i, k, t] - self.recovery_rate[k] * undetected_rcv[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            total_hospitalized[i, k, t + 1] >= self.ith_rate[k] * infected[i, k, t] * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )

        solver.addConstrs(
            deceased[i, k, t+1] - deceased[i, k, t] >= self.death_rate[k] * (
                    hospitalized_dth[i, k, t] + quarantine_dth[i, k, t] + undetected_dth[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            recovered[i, k, t + 1] - recovered[i, k, t] >= self.recovery_rate[k] * (
                    hospitalized_rcv[i, k, t] + quarantine_rcv[i, k, t] + undetected_rcv[i, k, t]
            ) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )

        # Set bounding constraint on infection variable
        solver.addConstrs(
            abs_box_infection[i, t] <= exploration_tol * infected_estimate[i, t]
            for i in self._regions for t in self._timesteps
        )
        solver.addConstrs(
            abs_box_infection[i, t] >= infected_estimate[i, t] - infected.sum(i, "*", t)
            for i in self._regions for t in self._timesteps
        )
        solver.addConstrs(
            abs_box_infection[i, t] >= - infected_estimate[i, t] + infected.sum(i, "*", t)
            for i in self._regions for t in self._timesteps
        )

        # Set resource constraints
        solver.addConstrs(
            vaccinated.sum("*", "*", t) == self.vaccine_budget[t]
            for t in self._planning_timesteps
        )
        solver.addConstrs(
            vaccinated.sum(i, "*", t) >= fairness_param * susceptible.sum(i, "*", t)
            for i in self._regions for t in self._planning_timesteps
        )

        # Set objective
        solver.setObjective(deceased.sum("*", "*", self._n_timesteps), GRB.MINIMIZE)

        solver.params.MIPGap = mip_gap
        solver.params.OutputFlag = output_flag
        solver.params.FeasibilityTol = feasibility_tol
        solver.params.TimeLimit = time_limit

        # Solve model
        solver.optimize()

        # Return total deceased and vaccine policy
        return self._get_variable_value(solver=solver, variable=deceased), \
            self._get_variable_value(solver=solver, variable=vaccinated)

    def solve_prescriptive_delphi(
            self,
            exploration_tol: float,
            termination_tol: float,
            num_restarts: int = 100
    ) -> Tuple[DiscreteDELPHISolution, list, Union[np.ndarray, None]]:

        # Initialize DiscreteDELPHISolution container
        best_delphi_solution = DiscreteDELPHISolution()

        # Initialize algorithm parameters
        cost_trajectories = []
        best_obj = np.inf
        optimal_vaccines = None

        # Iterate through each random restart
        for trial in range(num_restarts):

            current_delphi_solution = DiscreteDELPHISolution()
            obj_vals = []
            prev_obj = np.inf

            # Generate a random feasible solution and store objective value (#total deaths)
            vaccines, current_obj = self.generate_feasible_policy()
            obj_vals.append(current_obj)

            while prev_obj - current_obj >= termination_tol:

                # Solve DELPHI evolution given vaccine policy and update the DELPHISolution container
                state_variables = self.solve_stage1(vaccines)
                current_delphi_solution.functional_states = state_variables
                estimated_infections, _, _ = current_delphi_solution.functional_states

                new_obj, new_vaccines = self.solve_stage2(estimated_infections, exploration_tol)

                if new_obj < current_obj:
                    prev_obj, current_obj, vaccines = current_obj, new_obj, new_vaccines
                    obj_vals.append(current_obj)
                else:
                    break

            cost_trajectories.append({trial: obj_vals})

            # If the policy produced during the random trial is better than current one, update all parameters
            if current_obj < best_obj:
                best_obj = current_obj
                optimal_vaccines = vaccines
                best_delphi_solution.functional_states = self.solve_stage1(optimal_vaccines)

        return best_delphi_solution, cost_trajectories, optimal_vaccines

    def solve_prescriptive_delphi_with_annealing(
            self,
            exploration_tol: float,
            termination_tol: float,
            num_restarts: int,
            temperature: float
    ):
        # TODO : currently thinking about doing this in a smart way
        pass
