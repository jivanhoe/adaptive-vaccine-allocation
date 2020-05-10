from models.folding_horizon import FoldingHorizonAllocationModel
from models.proportional_allocation_model import solve_proportional_allocation_model
from models.nominal_model import solve_nominal_model
from models.robust_model import solve_robust_model
from models.robust_reformulated_model import solve_robust_reformulated_model
from utils.data_processing import load_data

import pandas as pd

# PATHS
DATA_DIR = "../../data/processed/"

# PARAMS
PLANNING_HORIZON = 5
NUM_TRIALS = 10
NOISE_GRID = [
    {"rep_factor_noise": 0.05, "morbidity_rate_noise": 0.1},
    {"rep_factor_noise": 0.1, "morbidity_rate_noise": 0.2},
    {"rep_factor_noise": 0.15, "morbidity_rate_noise": 0.3}
]
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
        solver_params={},
        planning_horizon=PLANNING_HORIZON,
        name="nominal"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_robust_model,
        solver_params={"sigma": 0.2, "delta": 0.05},
        planning_horizon=PLANNING_HORIZON,
        name="robust"
    ),
    FoldingHorizonAllocationModel(
        solver=solve_robust_reformulated_model,
        solver_params={"sigma": 0.2, "delta": 0.05},
        planning_horizon=PLANNING_HORIZON,
        name="robust_reformulation"
    ),
]

pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = load_data(data_dir=DATA_DIR)
rep_factor = rep_factor[:, 0]
budget[0] = 0

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
            num_trials=NUM_TRIALS,
            **noise
        )
        print(f"Completed trials for noise {noise} and model {allocation_model.name}")
        pd.DataFrame(results).to_csv("simulation-results-v2.csv")




