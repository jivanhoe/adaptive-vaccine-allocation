from typing import Tuple, NoReturn
import numpy as np
from src.models.delphi import HeuristicDELPHIModel


class RandomDELPHIData:

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
            initial_pct_hospitalized_range: Tuple[float, float] = (1e-2, 5e-2),
            initial_pct_quarantined_range: Tuple[float, float] = (0.1, 0.3),
            initial_pct_undetected_range: Tuple[float, float] = (0.2, 0.5),
            initial_pct_recovered_range: Tuple[float, float] = (0.01, 0.05),
            infection_rate_range: Tuple[float, float] = (0.1, 0.3),
            policy_response: Tuple[int, ...] = (0, 1),
            progression_rate_range: Tuple[float, float] = (0.05, 0.15),
            detection_rate_range: Tuple[float, float] = (0.25, 0.4),
            pct_infected_detected_range: Tuple[float, float] = (0.15, 0.25),
            pct_detected_hospitalized_range: Tuple[float, float] = (0.12, 0.18),
            recovery_rate_range: Tuple[float, float] = (0.1, 0.2),
            mortality_rate_range: Tuple[float, float] = (0.05, 0.1),
            rate_of_death_range: Tuple[float, float] = (0.1, 0.15)
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
        self.initial_pct_hospitalized_range = initial_pct_hospitalized_range
        self.initial_pct_quarantined_range = initial_pct_quarantined_range
        self.initial_pct_undetected_range = initial_pct_undetected_range
        self.initial_pct_recovered_range = initial_pct_recovered_range

        # Attributes for model parameters
        self.infection_rate_range = infection_rate_range
        self.policy_response = policy_response
        self.progression_rate_range = progression_rate_range
        self.detection_rate_range = detection_rate_range
        self.pct_infected_detected_range = pct_infected_detected_range
        self.pct_detected_hospitalized_range = pct_detected_hospitalized_range
        self.recovery_rate_range = recovery_rate_range
        self.mortality_rate_range = mortality_rate_range
        self.rate_of_death_range = rate_of_death_range

        self._validate_inputs()

    def _validate_inputs(self) -> NoReturn:
        pass

    @staticmethod
    def _uniform(min_val: float, max_val: float, dims: Tuple[int, ...]) -> np.ndarray:
        return (max_val - min_val) * np.random.rand(*dims) + min_val

    @staticmethod
    def _normal(mean: float, std: float, dims: Tuple[int, ...]) -> np.ndarray:
        return std * np.random.rand(*dims) + mean

    def _generate_initial_conditions(self):
        # Generate population subset sizes
        population = RandomDELPHIData._uniform(
            *self.population_subset_range,
            dims=(self.n_regions, self.n_risk_classes)
        )

        # Generate exposed, infected, hospitalized, quarantined, undetected, and recovered populations for each subset
        initial_exposed = RandomDELPHIData._uniform(
            *self.initial_pct_exposed_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        initial_infected = RandomDELPHIData._uniform(
            *self.initial_pct_infected_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        initial_hospitalized = RandomDELPHIData._uniform(
            *self.initial_pct_hospitalized_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * initial_infected

        initial_quarantined = RandomDELPHIData._uniform(
            *self.initial_pct_quarantined_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * initial_infected

        initial_undetected = RandomDELPHIData._uniform(
            *self.initial_pct_undetected_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * initial_infected

        initial_recovered = RandomDELPHIData._uniform(
            *self.initial_pct_recovered_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        # Infer susceptible population for subset
        initial_susceptible = population - initial_exposed - initial_infected - initial_hospitalized - \
            initial_quarantined - initial_undetected - initial_recovered

        return population, initial_susceptible, initial_exposed, initial_infected, initial_hospitalized, \
            initial_quarantined, initial_undetected, initial_recovered

    def _get_vaccine_budget(self, initial_susceptible: np.ndarray) -> np.ndarray:
        pass

    def _generate_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def generate_model(self, seed: int = 0) -> HeuristicDELPHIModel:
        # Set seed
        np.random.seed(seed)

        # Generate initial conditions
        initial_vars = self._generate_initial_conditions()

        # Get vaccine budget based on initial susceptible population
        vaccine_budget = self._get_vaccine_budget(initial_susceptible=initial_susceptible)

        # Generate parameters
        infection_rate, progression_rate, recovery_rate, death_rate = self._generate_parameters()

        initial_conditions = {
            'initial_susceptible': None,
            'initial_exposed': None,
            'initial_infected': None,
            'initial_hospitalized_dth': None,
            'initial_hospitalized_rcv': None,
            'initial_quarantine_dth': None,
            'initial_quarantine_rcv': None,
            'initial_undetected_dth': None,
            'initial_undetected_rcv': None,
            'initial_recovered': None,
            'population': None,
            'vaccine_budget': None,
            'other_vaccinated': None
        }

        model_parameters = {
            'infection_rate': None,
            'policy_response': None,
            'progression_rate': None,
            'ihd_rate': None,
            'ihr_rate': None,
            'iqd_rate': None,
            'iqr_rate': None,
            'iud_rate': None,
            'iur_rate': None,
            'ith_rate': None,
            'detection_rate': None,
            'death_rate': None,
            'recovery_rate': None,
            'days_per_timestep': None
        }

        return HeuristicDELPHIModel(
            init_variables=initial_conditions,
            params=model_parameters
        )
