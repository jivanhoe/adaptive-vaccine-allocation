from models.folding_horizon import FoldingHorizonAllocationModel
from models.proportional_allocation_model import solve_proportional_allocation_model
from models.nominal_model import solve_nominal_model
from models.robust_model import solve_robust_model
from models.robust2_model import solve_robust2_model
from utils.data_processing import get_toy_data_from_census_data
from utils.load_data import process_data

import pandas as pd
import os

# PATHS census data
POP_DATA_PATH = "../../data/census/pop-data.csv"
LAND_AREA_DATA_PATH = "../../data/census/land-area-data.csv"

# PATHS realized data
FOLDER_PATH = "/Users/alessandropreviero/Downloads/processed_data"
immunized_pop_data_path = os.path.join(FOLDER_PATH, "immunized_pop.csv")
pop_data_path = os.path.join(FOLDER_PATH, "pop.csv")
active_cases_data_path = os.path.join(FOLDER_PATH, "active_cases.csv")
rep_factor_data_path = os.path.join(FOLDER_PATH, "rep_factor.csv")
morbidity_data_path = os.path.join(FOLDER_PATH, "morbidity_rate.csv")

USE_CENSUS_DATA = False
USE_REAL_DATA = True

# PARAMS
PLANNING_HORIZON = 5
NUM_TRIALS = 1
NOISE_GRID = [0.05]
MODELS = [
    FoldingHorizonAllocationModel(
        solver=solve_proportional_allocation_model,
        solver_params={"with_prioritization": False},
        planning_horizon=PLANNING_HORIZON,
        name="proportional_allocation"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_proportional_allocation_model,
        solver_params={"with_prioritization": True},
        planning_horizon=PLANNING_HORIZON,
        name="proportional_allocation_with_prioritization"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_nominal_model,
        solver_params={"time_limit": 60, "mip_gap": 0.01, "output_flag": False},
        planning_horizon=PLANNING_HORIZON,
        name="nominal"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_robust_model,
        solver_params={"rho": 0.1, "gamma": 3.0, "delta": 0.1, "time_limit": 60, "mip_gap": 0.01, "output_flag": False},
        planning_horizon=PLANNING_HORIZON,
        name="robust"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_robust2_model,
        solver_params={"time_limit": 60, "mip_gap": 0.01, "output_flag": False},
        planning_horizon=PLANNING_HORIZON,
        name="robust2"
    )
]

if USE_CENSUS_DATA:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
        pop_data_path=POP_DATA_PATH,
        land_area_data_path=LAND_AREA_DATA_PATH,
        groupby_state=True,
    )

else:
    states, regions, pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = process_data(
        pop_data_path,
        rep_factor_data_path,
        morbidity_data_path,
        immunized_pop_data_path,
        active_cases_data_path,
        pct_budget=0.25)


results = []
for allocation_model in MODELS:
    for noise in NOISE_GRID:
        results += allocation_model.run_simulations(
            pop=pop,
            immunized_pop=immunized_pop,
            active_cases=active_cases,
            rep_factor=rep_factor,
            morbidity_rate=morbidity_rate,
            budget=budget,
            noise=noise,
            num_trials=NUM_TRIALS
        )
pd.DataFrame(results).to_csv("simulation_results.csv")




