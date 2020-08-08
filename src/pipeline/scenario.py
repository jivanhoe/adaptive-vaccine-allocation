from typing import Dict, Union, List, Tuple, Optional
import pickle

import pandas as pd

from pipeline.constants import *
from pipeline.data_loading import load_and_clean_delphi_predictions, load_and_clean_delphi_params
from pipeline.data_processing import calculate_n_timesteps, get_initial_conditions, get_delphi_params
from models.prescriptive_delphi_model import PrescriptiveDELPHIModel


class Scenario:

    def __init__(
            self,
            start_date: dt.datetime,
            end_date: dt.datetime,
            vaccine_effectiveness: float,
            daily_vaccine_budget: float,
            max_total_capacity: Optional[float] = None,
            max_allocation_factor: float = MAX_ALLOCATION_FACTOR,
            min_allocation_factor: float = MIN_ALLOCATION_FACTOR,
            max_increase_pct: float = MAX_INCREASE_PCT,
            max_decrease_pct: float = MAX_DECREASE_PCT,
            excluded_risk_classes: List[int] = EXCLUDED_RISK_CLASSES,
            optimize_capacity: bool = OPTIMIZE_CAPACITY
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.vaccine_effectiveness = vaccine_effectiveness
        self.daily_vaccine_budget = daily_vaccine_budget
        self.max_total_capacity = max_total_capacity if max_total_capacity else daily_vaccine_budget
        self.max_allocation_factor = max_allocation_factor
        self.min_allocation_factor = min_allocation_factor
        self.max_increase_pct = max_increase_pct
        self.max_decrease_pct = max_decrease_pct
        self.excluded_risk_classes = excluded_risk_classes
        self.optimize_capacity = optimize_capacity

    def get_vaccine_params(
            self,
            total_pop: float,
    ) -> Dict[str, Union[float, np.ndarray]]:
        n_timesteps = calculate_n_timesteps(start_date=self.start_date, end_date=self.end_date)
        return dict(
            vaccine_effectiveness=self.vaccine_effectiveness,
            vaccine_budget=np.ones(n_timesteps) * self.daily_vaccine_budget,
            max_total_capacity=self.max_total_capacity,
            max_allocation_pct=self.daily_vaccine_budget / total_pop * self.max_allocation_factor,
            min_allocation_pct=self.daily_vaccine_budget / total_pop * self.min_allocation_factor,
            max_decrease_pct=self.max_increase_pct,
            max_increase_pct=self.max_decrease_pct,
            excluded_risk_classes=np.array(self.excluded_risk_classes) if self.excluded_risk_classes else np.array([]).astype(int),
            optimize_capacity=self.optimize_capacity,
        )

    def load_model(
            self,
            mortality_rate_path: Optional[str] = None
    ) -> PrescriptiveDELPHIModel:

        # Load raw data
        params_df = load_and_clean_delphi_params(DELPHI_PARAMS_PATH)
        predictions_df = load_and_clean_delphi_predictions(DELPHI_PREDICTIONS_PATH)
        cdc_df = pd.read_csv(CDC_DATA_PATH)
        pop_df = pd.read_csv(POPULATION_DATA_PATH)

        # Get processed data for model
        initial_conditions = get_initial_conditions(
            pop_df=pop_df,
            predictions_df=predictions_df,
            start_date=self.start_date
        )
        delphi_params = get_delphi_params(
            pop_df=pop_df,
            cdc_df=cdc_df,
            params_df=params_df,
            predictions_df=predictions_df,
            start_date=self.start_date,
            end_date=self.end_date,
            mortality_rate_path=mortality_rate_path
        )
        vaccine_params = self.get_vaccine_params(total_pop=initial_conditions["population"].sum())

        # Return prescriptive DELPHI model object
        return PrescriptiveDELPHIModel(
            initial_conditions=initial_conditions,
            delphi_params=delphi_params,
            vaccine_params=vaccine_params
        )

    def run(
            self,
            model_path: str,
            baseline_solution_path: str,
            optimized_solution_path: str,
            mortality_rate_path: str,
            reload_mortality_rate: bool = False
    ) -> Tuple[float, float]:

        print("Loading model...")
        model = self.load_model(mortality_rate_path=mortality_rate_path if reload_mortality_rate else None)
        if not reload_mortality_rate:
            with open(mortality_rate_path, "wb") as fp:
                np.save(fp, model.mortality_rate)

        print("Running baseline...")
        baseline_solution = model.simulate(prioritization_allocation=False)
        with open(baseline_solution_path, "wb") as fp:
            pickle.dump(baseline_solution, fp)

        print("Optimizing...")
        optimized_solution = model.optimize(
            exploration_tol=EXPLORATION_TOL,
            termination_tol=TERMINATION_TOL,
            max_iterations=MAX_ITERATIONS,
            n_early_stopping_iterations=N_EARLY_STOPPING_ITERATIONS,
            barrier_conv_tol=BARRIER_CONV_TOL,
            log=True
        )
        with open(optimized_solution_path, "wb") as fp:
            pickle.dump(optimized_solution, fp)
        with open(model_path, "wb") as fp:
            pickle.dump(model, fp)

        return baseline_solution.get_objective_value(), optimized_solution.get_objective_value()

