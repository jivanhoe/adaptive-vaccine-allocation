from models.folding_horizon import FoldingHorizonAllocationModel
from models.proportional_allocation_model import solve_proportional_allocation_model
from models.nominal_model import solve_nominal_model
from models.robust_model import solve_robust_model
from utils.data_processing import get_toy_data_from_census_data

import pandas as pd


# PATHS
POP_DATA_PATH = "../../data/census/pop-data.csv"
LAND_AREA_DATA_PATH = "../../data/census/land-area-data.csv"

# PARAMS
PLANNING_HORIZON = 5
NUM_TRIALS = 10
NOISE_GRID = [0.05, 0.2]
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
        solver_params={"time_limit": 120, "mip_gap": 0.01, "output_flag": False},
        planning_horizon=PLANNING_HORIZON,
        name="nominal"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_nominal_model,
        solver_params={"time_limit": 120, "mip_gap": 0.01, "output_flag": False},
        planning_horizon=PLANNING_HORIZON,
        name="nominal"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_robust_model,
        solver_params={"rho": 0.1, "gamma": 3.0, "delta": 0.1, "time_limit": 120, "mip_gap": 0.01, "output_flag": False},
        planning_horizon=PLANNING_HORIZON,
        name="robust"
    ),
]

pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
    pop_data_path=POP_DATA_PATH,
    land_area_data_path=LAND_AREA_DATA_PATH,
    groupby_state=True,
)

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




