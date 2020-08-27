import pandas as pd
import os

from pipeline.scenario import Scenario
from pipeline.constants import *


if __name__ == "__main__":

    scenario_params_grid = [
        dict(
            **dates,
            vaccine_effectiveness=vaccine_effectiveness,
            daily_vaccine_budget=daily_vaccine_budget,
            min_allocation_factor=min_allocation_factor
        )
        for dates in DATES_GRID
        for vaccine_effectiveness in VACCINE_EFFECTIVENESS_GRID
        for daily_vaccine_budget in DAILY_VACCINE_BUDGET_GRID
        for min_allocation_factor in MIN_ALLOCATION_FACTOR_GRID
    ]

    for i, scenario_params in enumerate(scenario_params_grid):

        start_date = scenario_params["start_date"]
        end_date = scenario_params["start_date"]
        mortality_rate_path = f"{MORTALITY_RATES_PATH}{start_date}-{end_date}.npy"
        reload_mortality_rate = os.path.isfile(mortality_rate_path)

        baseline_obj_val, optimized_obj_val = Scenario(**scenario_params).run(
            model_path=f"{MODEL_PATH_PATH}{i}.pickle",
            baseline_solution_path=f"{BASELINE_SOLUTION_PATH}{i}.pickle",
            optimized_solution_path=f"{OPTIMIZED_SOLUTION_ATH}{i}.pickle",
            mortality_rate_path=mortality_rate_path,
            reload_mortality_rate=reload_mortality_rate
        )
        scenario_params["baseline_obj_val"] = baseline_obj_val
        scenario_params["optimized_obj_val"] = optimized_obj_val

    results = pd.DataFrame(scenario_params_grid)
    results["abs_improvement"] = results["baseline_obj_val"] - results["optimized_obj_val"]
    results["pct_improvement"] = results["abs_improvement"] / results["baseline_obj_val"] * 1e2
    results.to_csv(RESULTS_PATH)
