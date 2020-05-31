from typing import Tuple, Dict, Union

import numpy as np

from models.delphi import PrescriptiveDELPHIModel


class RandomDELPHIData:

    def __init__(
            self,
            n_regions: int,
            n_risk_classes: int,
            n_timesteps: int,
            planning_period: int = 1,
            days_per_timestep: float = 1.0,
            vaccine_budget_pct: float = 0.01,
            population_subset_range: Tuple[float, float] = (1e3, 5e3),
            initial_pct_exposed_range: Tuple[float, float] = (0.01, 0.05),
            initial_pct_infected_range: Tuple[float, float] = (0.01, 0.05),
            initial_pct_hospitalized_range: Tuple[float, float] = (0.0, 1e-4),
            initial_pct_quarantined_range: Tuple[float, float] = (0.0, 0.03),
            initial_pct_undetected_range: Tuple[float, float] = (0.0, 0.03),
            initial_pct_recovered_range: Tuple[float, float] = (0.01, 0.05),
            infection_rate_range: Tuple[float, float] = (0.1, 0.3),
            hospitalization_prob_range: Tuple[float, float] = (0, 0.3),
            death_prob_range: Tuple[float, float] = (1e-3, 0.1),
            detection_prob: float = 0.2,
            median_progression_time: float = 5.0,
            median_detection_time: float = 2.0,
            median_recovery_time: float = 15.0,
            median_death_time: float = 20.0
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
        self.hospitalization_prob = np.linspace(*hospitalization_prob_range, n_risk_classes)
        self.death_prob = np.linspace(*death_prob_range, n_risk_classes)
        self.detection_prob = detection_prob
        self.infection_rate_range = infection_rate_range
        self.median_progression_time = median_progression_time
        self.median_detection_time = median_detection_time
        self.median_recovery_time = median_recovery_time
        self.median_death_time = median_death_time

    @staticmethod
    def _uniform(min_val: float, max_val: float, dims: Tuple[int, ...]) -> np.ndarray:
        return (max_val - min_val) * np.random.rand(*dims) + min_val

    @staticmethod
    def _normal(mean: float, std: float, dims: Tuple[int, ...]) -> np.ndarray:
        return std * np.random.rand(*dims) + mean

    def _get_vaccine_budget(self, initial_susceptible: np.ndarray) -> np.ndarray:
        budget_per_period = initial_susceptible.sum() * self.vaccine_budget_pct
        return np.array([0 if t % self.planning_period else budget_per_period for t in range(self.n_timesteps)])

    def _generate_initial_conditions(self) -> Dict[str, np.ndarray]:

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

        initial_infectious = RandomDELPHIData._uniform(
            *self.initial_pct_infected_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        initial_hospitalized = RandomDELPHIData._uniform(
            *self.initial_pct_hospitalized_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        initial_quarantined = RandomDELPHIData._uniform(
            *self.initial_pct_quarantined_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        initial_undetected = RandomDELPHIData._uniform(
            *self.initial_pct_undetected_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        initial_recovered = RandomDELPHIData._uniform(
            *self.initial_pct_recovered_range,
            dims=(self.n_regions, self.n_risk_classes)
        ) * population

        # Infer susceptible population for subset
        initial_susceptible = population - initial_exposed - initial_infectious - initial_hospitalized - \
            initial_quarantined - initial_undetected - initial_recovered

        # Return all initial conditions
        death_prob = np.tile(self.death_prob, reps=(self.n_regions, 1))
        return {
            "initial_susceptible": initial_susceptible,
            "initial_exposed": initial_exposed,
            "initial_infectious": initial_infectious,
            "initial_hospitalized_dying": initial_hospitalized * death_prob,
            "initial_hospitalized_recovering": initial_hospitalized * (1 - death_prob),
            "initial_quarantined_dying": initial_quarantined * death_prob,
            "initial_quarantined_recovering": initial_quarantined * (1 - death_prob),
            "initial_undetected_dying": initial_undetected * death_prob,
            "initial_undetected_recovering": initial_undetected * (1 - death_prob),
            "initial_recovered": initial_recovered,
            "population": population.sum(axis=1),
            "vaccine_budget": self._get_vaccine_budget(initial_susceptible=initial_susceptible),
        }

    def _generate_parameters(self) -> Dict[str, Union[float, np.ndarray]]:

        # Randomly generate infection rates
        infection_rate = RandomDELPHIData._uniform(
            *self.infection_rate_range,
            dims=(self.n_regions,)
        )

        # Return all parameters
        detection_rate = np.log(2) / self.median_detection_time
        return {
            "infection_rate": infection_rate,
            "policy_response": np.ones((self.n_regions, self.n_timesteps)),
            "progression_rate": np.log(2) / self.median_death_time,
            "detection_rate": detection_rate,
            "ihd_transition_rate": detection_rate * self.detection_prob * self.hospitalization_prob * self.death_prob,
            "ihr_transition_rate": detection_rate * self.detection_prob * self.hospitalization_prob * (1 - self.death_prob),
            "iqd_transition_rate": detection_rate * self.detection_prob * (1 - self.hospitalization_prob) * self.death_prob,
            "iqr_transition_rate": detection_rate * self.detection_prob * (1 - self.hospitalization_prob) * (1 - self.death_prob),
            "iud_transition_rate": detection_rate * (1 - self.detection_prob) * self.death_prob,
            "iur_transition_rate": detection_rate * (1 - self.detection_prob) * self.detection_prob,
            "death_rate": np.log(2) / self.median_death_time,
            "recovery_rate": np.log(2) / self.median_recovery_time,
            "days_per_timestep": self.days_per_timestep
        }

    def generate_model(self, seed: int = 0) -> PrescriptiveDELPHIModel:

        # Set seed
        np.random.seed(seed)

        # Return DELPHI model with randomly generated initial conditions and parameters
        return PrescriptiveDELPHIModel(
            initial_conditions=self._generate_initial_conditions(),
            params=self._generate_parameters()
        )
