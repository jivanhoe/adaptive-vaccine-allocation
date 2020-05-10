from typing import Tuple
import numpy as np
import pandas as pd
import os


def to_array(df: pd.DataFrame) -> np.ndarray:

    if {'state', 'region'}.issubset(df.columns):
        df = df.drop(columns=["state", "region"])

    return np.array(df)


def process_data(
        pop_data_path: str,
        rep_factor_data_path: str,
        morbidity_data_path: str,
        immunized_data_path: str,
        active_cases_data_path: str,
        pct_budget: float = 1e-1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    pop = pd.read_csv(pop_data_path)
    states = np.array(pop.state)
    regions = np.array(pop.region)

    pop = to_array(pop)
    rep_factor = to_array(pd.read_csv(rep_factor_data_path)).flatten()
    morbidity_rate = to_array(pd.read_csv(morbidity_data_path))
    immunized_pop = to_array(pd.read_csv(immunized_data_path))
    active_cases = to_array(pd.read_csv(active_cases_data_path))
    budget = np.ones(int(np.round(1 / pct_budget))) * (pop - immunized_pop - active_cases).sum() * pct_budget
    budget[0] = 0

    return states, regions, pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget


