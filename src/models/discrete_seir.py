from typing import Optional, Tuple

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB


class DiscreteSEIRSolution:

    def __init__(
            self,
            susceptible: np.ndarray,
            exposed: np.ndarray,
            infected: np.ndarray,
            recovered: np.ndarray,
            deceased: np.ndarray,
            vaccinated: np.ndarray,
            days_per_timestep: float,
            validate_on_init: bool = False
    ):
        """
        Instantiate a container object for a discrete SEIR solution that allows for easy querying and plotting.
        :param susceptible: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1)
        :param exposed: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1)
        :param infected: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1)
        :param recovered: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1)
        :param deceased: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1)
        :param vaccinated: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1)
        :param days_per_timestep: a float that specifies the step size of the discretization scheme used to compute the
            solution
        :param validate_on_init: a boolean that specifies whether to perform automatic validation checks on the solution
        """
        self.susceptible = susceptible
        self.exposed = exposed
        self.infected = infected
        self.recovered = recovered
        self.deceased = np.zeros(susceptible.shape) if deceased is None else deceased
        self.vaccinated = np.zeros(susceptible.shape) if vaccinated is None else vaccinated
        self.days_per_timestep = days_per_timestep

        # Check solution
        if validate_on_init:
            self._validate_solution()

    def _validate_solution(self) -> None:
        """
        Check that the provided solution arrays have valid dimensions and values.
        :return: None
        """
        expected_dims = self.susceptible.shape
        for name, array in [
            ("susceptible", self.susceptible),
            ("exposed", self.exposed),
            ("infected", self.infected),
            ("recovered", self.recovered),
            ("deceased", self.deceased),
            ("vaccinated", self.vaccinated),
        ]:
            assert array.shape == expected_dims, \
                f"Invalid dimensions for {name} array - expected {expected_dims}, received {array.shape}"
            assert np.all(array >= 0), f"Invalid {name} array - all values must be non-negative"

    def get_total_deaths(self) -> float:
        return self.deceased[:, :, -1].sum()

    def get_total_infections(self) -> float:
        infected = self.infected.sum(axis=(0, 1))
        return 0.5 * (infected[1:] + infected[:-1]).sum() * self.days_per_timestep

    def plot(self, figsize: Tuple[float, float] = (15.0, 7.5)) -> plt.figure:
        """
        Plot a visualization of the solution showing the change in population composition over time and the cumulative
        number of infections and deaths.
        :param figsize: A tuple o that specifies the dimension of the plot
        :return: a matplotlib figure object
        """

        # Initialize figure
        fig, ax = plt.subplots(ncols=2, figsize=figsize)

        # Define plot settings
        plot_settings = dict(alpha=0.7, linestyle="solid", marker="" if self.days_per_timestep < 1 else ".")

        # Get x-axis for plots
        days = np.arange(self.susceptible.shape[-1]) * self.days_per_timestep

        # Make plot of population breakdown
        total_pop = (self.susceptible[:, :, 0] + self.exposed[:, :, 0] + self.infected[:, :, 0] +
                     self.recovered[:, :, 0]).sum()
        ax[0].plot(
            days, self.susceptible.sum(axis=(0, 1)) / total_pop * 100,
            label="Susceptible", color="tab:blue", **plot_settings
        )
        ax[0].plot(
            days, (self.vaccinated.sum(axis=(0, 1)).cumsum() + self.recovered.sum(axis=(0, 1))) / total_pop * 100,
            label="Vaccinated or recovered", color="tab:green", **plot_settings
        )
        ax[0].plot(
            days, (self.exposed + self.infected).sum(axis=(0, 1)) / total_pop * 100,
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
        intfected = self.infected.sum(axis=(0, 1))
        ax[1].plot(
            days[:-1], (0.5 * (intfected[1:] + intfected[:-1]) * self.days_per_timestep).cumsum(),
            label="Infections", color="tab:orange", **plot_settings
        )
        ax[1].plot(
            days, self.deceased.sum(axis=(0, 1)),
            label="Deaths", color="black", **plot_settings
        )
        ax[1].legend(fontsize=12)
        ax[1].set_xlabel("Days", fontsize=14)
        ax[1].set_ylabel("Cumulative total", fontsize=14)
        ax[1].set_title("Casualties", fontsize=16)


class DiscreteSEIRModel:

    def __init__(
            self,
            initial_susceptible: np.ndarray,
            initial_exposed: np.ndarray,
            initial_infected: np.ndarray,
            initial_recovered: np.ndarray,
            vaccine_budget: np.ndarray,
            infection_rate: np.ndarray,
            progression_rate: np.ndarray,
            recovery_rate: np.ndarray,
            death_rate: np.ndarray,
            days_per_timestep: float,
    ):
        """
        Instantiate a discrete SEIR model with initial conditions and parameter estimates.
        :param initial_susceptible: a numpy array of size (n_regions, n_risk_classes) with the initial number of
            susceptible individuals in each population subset
        :param initial_exposed: a numpy array of size (n_regions, n_risk_classes) with the initial number of exposed
            individuals in each population subset
        :param initial_infected: a numpy array of size (n_regions, n_risk_classes) with the initial number of infected
            individuals in each population subset
        :param initial_recovered: a numpy array of size (n_regions, n_risk_classes) with the initial number of recovered
            individuals in each population subset
        :param vaccine_budget: a numpy array of size (n_timesteps,) that specifies the total number of vaccines to be
            allocated at each timestep
        :param infection_rate: a numpy array of size (n_regions, n_timesteps) with estimated infection rate by region at
            each timestep, in units of 1/days
        :param progression_rate: a numpy array of size (n_risk_classes,) with estimated progression rate by risk class,
            in units of 1/days
        :param recovery_rate: a numpy array of size (n_risk_classes,) with estimated recovery rate by risk class, in
            units of 1/days
        :param death_rate: a numpy array of size (n_risk_classes,) with estimated death rate by risk class, in units of
            1/days
        :param vaccine_budget: a numpy array of size (n_planning_periods,) that specifies the total number of vaccines
            to be allocated across each planning period
        :param days_per_timestep: a float that specifies the number of days in each timesteps
        """

        # Set initial conditions
        self.initial_susceptible = initial_susceptible
        self.initial_exposed = initial_exposed
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered
        self.vaccine_budget = vaccine_budget
        self.population = (initial_susceptible + initial_exposed + initial_infected + initial_recovered).sum(1)

        # Set model parameters
        self.infection_rate = infection_rate
        self.progression_rate = progression_rate
        self.death_rate = death_rate
        self.recovery_rate = recovery_rate
        self.days_per_timestep = days_per_timestep

        # Initialize helper attributes
        self._n_regions = self.initial_susceptible.shape[0]
        self._n_risk_classes = self.initial_susceptible.shape[1]
        self._n_timesteps = self.vaccine_budget.shape[0]
        self._regions = range(self._n_regions)
        self._risk_classes = range(self._n_risk_classes)
        self._timesteps = range(self._n_timesteps)
        self._planning_timesteps = [t for t in self._timesteps if self.vaccine_budget[t] > 0]
        self._non_planning_timesteps = [t for t in self._timesteps if self.vaccine_budget[t] == 0]

        # Perform validation checks
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Check that model inputs have valid dimensions and values.
        :return: None
        """

        for name, array, expected_dims in [
            ("initial susceptible", self.initial_susceptible, (self._n_regions, self._n_risk_classes)),
            ("initial exposed", self.initial_exposed, (self._n_regions, self._n_risk_classes)),
            ("initial infected", self.initial_infected, (self._n_regions, self._n_risk_classes)),
            ("initial recovered", self.initial_recovered, (self._n_regions, self._n_risk_classes)),
            ("infection rate", self.infection_rate, (self._n_regions, self._n_timesteps)),
            ("progression rate", self.progression_rate, (self._n_risk_classes,)),
            ("death rate", self.death_rate, (self._n_risk_classes,)),
            ("recovery rate", self.recovery_rate, (self._n_risk_classes,))
        ]:
            # Check dimensions match expected
            assert array.shape == expected_dims, \
                f"Invalid size for {name} array - expected {expected_dims}, received {array.shape}"

            # Check all values are non-negative
            assert np.all(array >= 0), f"Invalid {name} array - all values must be non-negative"

    def solve_baseline(self, prioritize_allocation: bool = True) -> DiscreteSEIRSolution:
        """
        Get solution under SEIR dynamics with baseline allocation heuristic.
        :param prioritize_allocation: a boolean argument that specifies whether high-risk individuals should be
            prioritized within each region under the allocation heuristic (default True)
        :return: a DiscreteSEIRSolution object
        """

        # Initialize arrays
        dims = (self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        susceptible = np.zeros(dims)
        exposed = np.zeros(dims)
        infected = np.zeros(dims)
        recovered = np.zeros(dims)
        deceased = np.zeros(dims)
        vaccinated = np.zeros(dims)

        # Set initial conditions
        susceptible[:, :, 0] = self.initial_susceptible
        exposed[:, :, 0] = self.initial_exposed
        infected[:, :, 0] = self.initial_infected
        recovered[:, :, 0] = self.initial_recovered

        # Propagate discrete SEIR dynamics with vaccine allocation heuristic
        for t in self._timesteps:

            # Check if total susceptible population is non-zero
            total_susceptible = susceptible[:, :, t].sum()
            if total_susceptible > 0:

                # If prioritized allocation is enabled, order region-wise allocation by risk class
                if prioritize_allocation:
                    regional_budget = susceptible[:, :, t].sum(1) / total_susceptible * self.vaccine_budget[t]
                    for k in np.argsort(-self.death_rate):
                        vaccinated[:, k, t] = np.minimum(regional_budget, susceptible[:, k, t])
                        regional_budget -= vaccinated[:, k, t]

                # Else allocate proportionally to each population subset
                else:
                    vaccinated[:, :, t] = np.minimum(
                        susceptible[:, :, t] / total_susceptible * self.vaccine_budget[t],
                        susceptible[:, :, t]
                    )

            # Apply forward difference scheme for each region
            for i in self._regions:
                susceptible[i, :, t + 1] = susceptible[i, :, t] + (
                        - self.infection_rate[i, t] / self.population[i] * (
                        susceptible[i, :, t] - vaccinated[i, :, t]) * infected[i, :, t].sum()
                ) * self.days_per_timestep - vaccinated[i, :, t]
                exposed[i, :, t + 1] = exposed[i, :, t] + (
                        self.infection_rate[i, t] / self.population[i] * (
                        susceptible[i, :, t] - vaccinated[i, :, t]) * infected[i, :, t].sum()
                        - self.progression_rate * exposed[i, :, t]
                ) * self.days_per_timestep
                infected[i, :, t + 1] = infected[i, :, t] + (
                        self.progression_rate * exposed[i, :, t]
                        - (self.recovery_rate + self.death_rate) * infected[i, :, t]
                ) * self.days_per_timestep
                recovered[i, :, t + 1] = recovered[i, :, t] + self.recovery_rate * infected[i, :, t] * self.days_per_timestep
                deceased[i, :, t + 1] = deceased[i, :, t] + self.death_rate * infected[i, :, t] * self.days_per_timestep

            # Clip negative values caused by discretization error
            exposed[:, :, t + 1] = np.maximum(exposed[:, :, t + 1], 0)
            infected[:, :, t + 1] = np.maximum(infected[:, :, t + 1], 0)

        # Return solution object
        return DiscreteSEIRSolution(
            susceptible=susceptible,
            exposed=exposed,
            infected=infected,
            recovered=recovered,
            deceased=deceased,
            vaccinated=vaccinated,
            days_per_timestep=self.days_per_timestep
        )

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

    def solve(
            self,
            fairness_param: float = 0.0,
            mip_gap: float = 1e-2,
            feasibility_tol: float = 1e-2,
            output_flag: bool = False,
            time_limit: float = 120.0
    ) -> DiscreteSEIRSolution:
        """
        Solve
        :param fairness_param: a float that specifies the minimum proportion of the susceptible population in each
            region that must be allocated a vaccine (default 0)
        :param mip_gap: a float that specifies the maximum MIP gap required for termination (default 1e-2)
        :param feasibility_tol: a float that specifies that maximum feasibility tolerance for constraints (default 1e-2)
        :param output_flag: a boolean that specifies whether to show the solver logs (default False)
        :param time_limit: a float that specifies the maximum solve time in seconds (default 120.0)
        :return: a DiscreteSEIRSolution object
        """

        # Initialize solver
        solver = gp.Model("SEIR")

        # Define variables
        susceptible = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        exposed = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        infected = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1)
        recovered = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        deceased = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)
        vaccinated = solver.addVars(self._n_regions, self._n_risk_classes, self._n_timesteps + 1, lb=0)

        # Set initial conditions
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
            recovered[i, k, 0] == self.initial_recovered[i, k]
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            deceased[i, k, 0] == 0
            for i in self._regions for k in self._risk_classes
        )
        solver.addConstrs(
            vaccinated[i, k, t] == 0
            for i in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )

        # Set constraints for SEIR dynamics
        solver.addConstrs(
            susceptible[i, k, t + 1] == susceptible[i, k, t] - vaccinated[i, k, t] - self.infection_rate[i, t] /
            self.population[i] * (susceptible[i, k, t] - vaccinated[i, k, t]) * infected.sum(i, "*", t)
            * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            susceptible[i, k, t + 1] == susceptible[i, k, t] - self.infection_rate[i, t] / self.population[i]
            * susceptible[i, k, t] * infected.sum(i, "*", t) * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            exposed[i, k, t + 1] == exposed[i, k, t] - susceptible[i, k, t + 1] + susceptible[i, k, t]
            - vaccinated[i, k, t] - self.progression_rate[k] * exposed[i, k, t] * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._planning_timesteps
        )
        solver.addConstrs(
            exposed[i, k, t + 1] == exposed[i, k, t] - susceptible[i, k, t + 1] + susceptible[i, k, t]
            - self.progression_rate[k] * exposed[i, k, t] * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._non_planning_timesteps
        )
        solver.addConstrs(
            infected[i, k, t + 1] == infected[i, k, t] + self.progression_rate[k] * exposed[i, k, t] * self.days_per_timestep
            - (self.recovery_rate[k] + self.death_rate[k]) * infected[i, k, t] * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            recovered[i, k, t + 1] >= recovered[i, k, t] + self.recovery_rate[k] * infected[i, k, t] * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
        )
        solver.addConstrs(
            deceased[i, k, t + 1] >= deceased[i, k, t] + self.death_rate[k] * infected[i, k, t] * self.days_per_timestep
            for i in self._regions for k in self._risk_classes for t in self._timesteps
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

        # Provide feasible warm start
        warm_start = self.solve_baseline(prioritize_allocation=True).vaccinated
        for i in self._regions:
            for k in self._risk_classes:
                for t in self._timesteps:
                    vaccinated[i, k, t].start = warm_start[i, k, t]

        # Set solver parameters
        solver.params.NonConvex = 2
        solver.params.MIPGap = mip_gap
        solver.params.OutputFlag = output_flag
        solver.params.FeasibilityTol = feasibility_tol
        solver.params.TimeLimit = time_limit

        # Solve model
        solver.optimize()

        # Return DiscreteSEIRSolution object
        return DiscreteSEIRSolution(
            susceptible=self._get_variable_value(solver=solver, variable=susceptible),
            exposed=self._get_variable_value(solver=solver, variable=exposed),
            infected=self._get_variable_value(solver=solver, variable=infected),
            recovered=self._get_variable_value(solver=solver, variable=recovered),
            vaccinated=self._get_variable_value(solver=solver, variable=vaccinated),
            deceased=self._get_variable_value(solver=solver, variable=deceased),
            days_per_timestep=self.days_per_timestep
        )


class RandomSEIRData:

    def __init__(
            self,
            n_regions: int,
            n_risk_classes: int,
            n_timesteps: int,
            planning_period: int,
            days_per_timestep: float = 1.0,
            vaccine_budget_pct: float = 0.1,
            population_subset_range: Tuple[float, float] = (1e3, 5e3),
            initial_pct_exposed_range: Tuple[float, float] = (0.01, 0.05),
            initial_pct_infected_range: Tuple[float, float] = (0.01, 0.05),
            initial_pct_recovered_range: Tuple[float, float] = (0.01, 0.05),
            infection_rate_range: Tuple[float, float] = (0.1, 0.3),
            progression_rate_range: Tuple[float, float] = (0.05, 0.15),
            recovery_rate_range: Tuple[float, float] = (0.1, 0.2),
            death_rate_range: Tuple[float, float] = (0.05, 0.1)
    ):

        # Attributes for problem size and difficulty
        self.n_regions = n_regions
        self.n_risk_classes = n_risk_classes
        self.n_timesteps = n_timesteps
        self.planning_period = planning_period
        self.days_per_timestep = days_per_timestep
        self.vaccine_budget_pct = vaccine_budget_pct

        # Attributes for initial conditions
        self.population_subset_range = population_subset_range
        self.initial_pct_exposed_range = initial_pct_exposed_range
        self.initial_pct_infected_range = initial_pct_infected_range
        self.initial_pct_recovered_range = initial_pct_recovered_range

        # Attributes for model parameters
        self.infection_rate_range = infection_rate_range
        self.progression_rate_range = progression_rate_range
        self.recovery_rate_range = recovery_rate_range
        self.death_rate_range = death_rate_range

        # Perform validation checks
        self._validate_inputs()

    def _validate_inputs(self) -> None:

        # Validate ranges independently
        for name, range_ in [
            ("population subset", self.progression_rate_range),
            ("initial % exposed", self.initial_pct_exposed_range),
            ("initial % infected", self.initial_pct_infected_range),
            ("initial % recovered", self.initial_pct_recovered_range),
            ("infection rate", self.infection_rate_range),
            ("progression rate", self.progression_rate_range),
            ("infection rate", self.infection_rate_range),
            ("infection rate", self.infection_rate_range),
        ]:
            assert range_[0] <= range_[1], f"Invalid {name} range - min must be less than or equal to max."
            assert range_[0] >= 0, f"Invalid {name} range - min must be non-negative."

        # Validate initial condition ranges
        assert self.initial_pct_exposed_range[1] + self.initial_pct_infected_range[1] + \
            self.initial_pct_recovered_range[1] < 1, \
            "Invalid ranges initial conditions - max proportion of exposed, infected and recovered must be less than 1."

        # Validate vaccine budget percentage
        assert 0 <= self.vaccine_budget_pct <= 1, "Invalid vaccine budget % - must be between 0 and 1"

    @staticmethod
    def _uniform(min_val: float, max_val: float, dims: Tuple[int, ...]) -> np.ndarray:
        return (max_val - min_val) * np.random.rand(*dims) + min_val

    @staticmethod
    def _normal(mean: float, std: float, dims: Tuple[int, ...]) -> np.ndarray:
        return std * np.random.rand(*dims) + mean

    def _generate_initial_conditions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Generate population subset sizes
        population = RandomSEIRData._uniform(
            *self.population_subset_range,
            dims=(self.n_regions, self.n_risk_classes)
        )

        # Generate exposed, infected and recovered populations for each subset
        initial_exposed = RandomSEIRData._uniform(
            *self.initial_pct_exposed_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population
        initial_infected = RandomSEIRData._uniform(
            *self.initial_pct_infected_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population
        initial_recovered = RandomSEIRData._uniform(
            *self.initial_pct_recovered_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        # Infer susceptible population for subset
        initial_susceptible = population - initial_exposed - initial_infected - initial_recovered

        # Return initial conditions
        return initial_susceptible, initial_exposed, initial_infected, initial_recovered

    def _get_vaccine_budget(self, initial_susceptible: np.ndarray) -> np.ndarray:
        budget_per_period = initial_susceptible.sum() * self.vaccine_budget_pct
        n_timesteps_required = np.ceil(self.planning_period / self.vaccine_budget_pct)
        return np.array([
            0 if (t % self.planning_period or t > n_timesteps_required) else budget_per_period
            for t in range(self.n_timesteps)
        ])

    def _generate_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Generate uniformly distributed infection infection rates by region and repeat for each timestep
        infection_rate = RandomSEIRData._uniform(
            *self.infection_rate_range,
            dims=(self.n_regions,)
        )
        infection_rate = np.tile(infection_rate, reps=(self.n_timesteps, 1)).T

        # Generate uniformly distributed death rates and sort in descending order
        progression_rate = RandomSEIRData._uniform(
            *self.progression_rate_range,
            dims=(self.n_risk_classes,)
        )
        progression_rate = np.sort(progression_rate)[::-1]

        # Generate uniformly distributed death rates and sort in ascending order
        recovery_rate = RandomSEIRData._uniform(
            *self.recovery_rate_range,
            dims=(self.n_risk_classes,)
        )
        recovery_rate = np.sort(recovery_rate)

        # Generate uniformly distributed death rates and sort in descending order
        death_rate = RandomSEIRData._uniform(
            *self.death_rate_range,
            dims=(self.n_risk_classes,)
        )
        death_rate = np.sort(death_rate)[::-1]

        # Return parameters
        return infection_rate, progression_rate, recovery_rate, death_rate

    def generate_model(self, seed: int = 0) -> DiscreteSEIRModel:
        # Set seed
        np.random.seed(seed)

        # Generate initial conditions
        initial_susceptible, initial_exposed, initial_infected, initial_recovered = self._generate_initial_conditions()

        # Get vaccine budget based on initial susceptible population
        vaccine_budget = self._get_vaccine_budget(initial_susceptible=initial_susceptible)

        # Generate parameters
        infection_rate, progression_rate, recovery_rate, death_rate = self._generate_parameters()

        # Return a DiscreteSEIRModel object
        return DiscreteSEIRModel(
            initial_susceptible=initial_susceptible,
            initial_exposed=initial_exposed,
            initial_infected=initial_infected,
            initial_recovered=initial_recovered,
            infection_rate=infection_rate,
            progression_rate=progression_rate,
            recovery_rate=recovery_rate,
            death_rate=death_rate,
            vaccine_budget=vaccine_budget,
            days_per_timestep=self.days_per_timestep
        )
