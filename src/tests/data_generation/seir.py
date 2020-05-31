from typing import Tuple
import numpy as np
from src.models.seir import DiscreteSEIRModel


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