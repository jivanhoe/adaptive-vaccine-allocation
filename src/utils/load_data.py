from typing import Tuple
import numpy as np
import pandas as pd
import os


def process_data(
        pop_data_path: str,
        rep_factor_data_path: str,
        morbidity_data_path: str,
        immunized_data_path: str,
        active_cases_data_path: str,
        pct_budget: float = 1e-1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def ToArray(df: pd.DataFrame) -> np.ndarray:
        colnames = df.columns
        if {'state', 'region'}.issubset(colnames):
            df = df.drop(columns=["state", "region"])

        return np.array(df)

    pop = pd.read_csv(pop_data_path)
    states = np.array(pop.state)
    regions = np.array(pop.region)

    pop = ToArray(pop)
    rep_factor = ToArray(pd.read_csv(rep_factor_data_path)).flatten()
    morbidity_rate = ToArray(pd.read_csv(morbidity_data_path))
    immunized_pop = ToArray(pd.read_csv(immunized_data_path))
    active_cases = ToArray(pd.read_csv(active_cases_data_path))
    budget = np.ones(int(np.round(1 / pct_budget))) * (pop - immunized_pop - active_cases).sum() * pct_budget
    budget[0] = 0

    return states, regions, pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget


