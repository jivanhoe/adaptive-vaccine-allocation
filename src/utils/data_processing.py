from copy import deepcopy
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd


def get_initial_conditions(
        pop_df: pd.DataFrame,
        param_df: pd.DataFrame,
        risk_classes: List[Dict[str, any]],
        n_regions: int,
        detection_prob: int,
        initial_recovery_ratio: float = 5.0
) -> Dict[str, np.ndarray]:
    """
    Get the initial conditions for prescriptive DELPHI model from raw data.

    :param pop_df: dataframe of population data
    :param param_df: dataframe of epidemiological parameters for DELPHI model fitted on historical data
    :param risk_classes: list of risk classes by sex and age group
    :param n_regions: integer value that specifies the number of regions into the population is partitioned
    :param detection_prob: float value that specifies that probability a case is detected by testing
    :param initial_recovery_ratio: a float value that specifies the ratio between the initial number of recovered
            and deceased population (default 5.0)
    :return: a dictionary of initial condition arrays
    """

    # Initialize arrays
    n_risk_classes = len(risk_classes)
    initial_default = np.zeros((n_regions, n_risk_classes))
    population = deepcopy(initial_default)
    initial_exposed = deepcopy(initial_default)
    initial_infectious = deepcopy(initial_default)

    param_df["initial_active_cases"] = param_df["initial_cases"] - (1 + initial_recovery_ratio) * param_df["initial_deaths"]

    for j, state in enumerate(pop_df["state"].unique()):
        for k, risk_class in enumerate(risk_classes):

            # Calculate total population for state and risk class
            population[j, k] = pop_df[
                pop_df["sex"].isin(risk_class["sexes"])
                & (pop_df["min_age"] >= risk_class["min_age"])
                & (pop_df["max_age"] <= risk_class["max_age"])
                & (pop_df["state"] == state)
            ]["population"].sum() / 1e3  # convert population units of thousands

        # Calculate initial exposed and infectious population for state
        initial_exposed[j, :] = population[j, :] / population[j, :].sum() / detection_prob \
            * param_df.loc[state, "pct_exposed"] * param_df.loc[state, "initial_active_cases"] / 1e3
        initial_infectious[j, :] = population[j, :] / population[j, :].sum() / detection_prob \
            * param_df.loc[state, "pct_infectious"] * param_df.loc[state, "initial_active_cases"] / 1e3

    # Return dictionary with all initial conditions
    return dict(
        initial_susceptible=population - initial_exposed - initial_infectious,
        initial_exposed=initial_exposed,
        initial_infectious=initial_infectious,
        initial_hospitalized_dying=initial_default,
        initial_hospitalized_recovering=initial_default,
        initial_quarantined_dying=initial_default,
        initial_quarantined_recovering=initial_default,
        initial_undetected_dying=initial_default,
        initial_undetected_recovering=initial_default,
        initial_recovered=initial_default,
        population=population
    )


def get_delphi_params(
        pop_df: pd.DataFrame,
        param_df: pd.DataFrame,
        mortality_df: pd.DataFrame,
        risk_classes: List[Dict[str, any]],
        n_regions: int,
        n_timesteps: int,
        detection_prob: float = 0.2,
        hospitalization_prob: float = 0.15,
        median_progression_time: float = 5.0,
        median_detection_time: float = 2.0,
        median_death_time: float = 20.0,
        median_recovery_time: float = 15.0,
        days_per_timestep: float = 1.0
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Get the DELPHI model parameters for prescriptive DELPHI model from raw data.

    :param pop_df: dataframe of population data
    :param param_df: dataframe of epidemiological parameters for DELPHI model fitted on historical data
    :param mortality_df: a dataframe of estimated mortality rates by sex and age group
    :param risk_classes: list of risk classes by sex and age group
    :param n_regions: integer value that specifies the number of regions into the population is partitioned
    :param n_timesteps: integer value that specifies the number of timesteps for which the dynamics are simulated
    :param detection_prob: float value in [0, 1] that specifies the probability a case is detected by testing
        (default 0.2)
    :param hospitalization_prob: float value in [0, 1] that specifies the probability a detected individual will be
        hospitalized (default 0.15)
    :param median_progression_time: positive float value that specifies the median time in days to transition from
        exposed to infectious (default 5.0)
    :param median_detection_time: positive float value that specifies the median time in days to transition from
        infectious to hospitalized, quarantined or undetected (default 2.0)
    :param median_death_time: positive float value that specifies the median time in days to transition from any dying
        state to deceased (default 20.0)
    :param median_recovery_time: positive float value that specifies the median time in days to transition from any
        recovering state to recovered (default 15.0)
    :param days_per_timestep: positive float value that specifies the number of days per timestep in the discretization
        scheme (default 1.0)
    :return: a dictionary of DELPHI model parameters
    """

    # Compute policy response for each state and timestep
    policy_response = np.ones((n_regions, n_timesteps))
    for j, state in enumerate(pop_df["state"].unique()):
        for t in range(n_timesteps):
            policy_response[j, t] = 2 / np.pi * np.arctan(
                -(t - param_df.loc[state]["intervention_start"]) * param_df.loc[state]["intervention_rate"] / 20) + 1

    # Compute mortality rate for each risk class
    n_risk_classes = len(risk_classes)
    mortality_rate = np.ones(n_risk_classes)
    for k, risk_class in enumerate(risk_classes):
        # TODO: make this smarter
        mortality_rate[k] = mortality_df[
            mortality_df["sex"].isin(risk_class["sexes"])
            & (mortality_df["min_age"] >= risk_class["min_age"])
            & (mortality_df["max_age"] <= risk_class["max_age"])
        ]["mortality_rate"].mean()

        # Return dictionary with all model parameters
    detection_rate = np.log(2) / median_detection_time
    hospitalization_prob = mortality_rate / mortality_rate.mean() * hospitalization_prob
    return dict(
        infection_rate=np.array(param_df["infection_rate"]),
        policy_response=policy_response,
        progression_rate=np.log(2) / median_progression_time,
        detection_rate=detection_rate,
        ihd_transition_rate=detection_rate * detection_prob * hospitalization_prob * mortality_rate,
        ihr_transition_rate=detection_rate * detection_prob * hospitalization_prob * (1 - mortality_rate),
        iqd_transition_rate=detection_rate * detection_prob * (1 - hospitalization_prob) * mortality_rate,
        iqr_transition_rate=detection_rate * detection_prob * (1 - hospitalization_prob) * (1 - mortality_rate),
        iud_transition_rate=detection_rate * (1 - detection_prob) * mortality_rate,
        iur_transition_rate=detection_rate * (1 - detection_prob) * (1 - mortality_rate),
        death_rate=np.log(2) / median_death_time,
        recovery_rate=np.log(2) / median_recovery_time,
        mortality_rate=mortality_rate,
        days_per_timestep=days_per_timestep
    )


def get_vaccine_params(
        n_timesteps: int,
        total_pop: float,
        vaccine_budget_pct: float,
        vaccine_effectiveness: float,
        max_allocation_pct: float,
        min_allocation_pct: float,
        max_total_capacity_pct: Optional[float] = None,
        optimize_capacity: bool = False,
        excluded_risk_classes: Optional[List[int]] = None,
        planning_period: int = 1
) -> Dict[str, Union[float, np.ndarray]]:
    """

    :param n_timesteps:
    :param total_pop:
    :param vaccine_budget_pct:
    :param vaccine_effectiveness: positive float value in [0, 1] that specifies the probability a vaccinated individual
        will become immune (default 0.5)
    :param max_allocation_pct:
    :param min_allocation_pct:
    :param max_total_capacity_pct:
    :param optimize_capacity:
    :param excluded_risk_classes:
    :param planning_period:
    :return:
    """
    return dict(
        vaccine_budget=np.array([total_pop * vaccine_budget_pct for t in range(n_timesteps) if not t % planning_period]),
        vaccine_effectiveness=vaccine_effectiveness,
        max_allocation_pct=max_allocation_pct,
        min_allocation_pct=min_allocation_pct,
        max_total_capacity=(max_total_capacity_pct if max_total_capacity_pct else vaccine_budget_pct) * total_pop,
        optimize_capacity=optimize_capacity,
        excluded_risk_classes=np.array(excluded_risk_classes) if excluded_risk_classes else np.array([]).astype(int),
    )
