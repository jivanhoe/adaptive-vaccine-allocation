from experiments.folding_horizon import solve_folding_horizon
from models.nominal_model import solve_nominal_model
from models.proportional_allocation_model import solve_proportional_allocation_model
from utils.data_processing import get_toy_data_from_census_data

POP_DATA_PATH = "../../data/census/pop-data.csv"
LAND_AREA_DATA_PATH = "../../data/census/land-area-data.csv"

pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
    pop_data_path=POP_DATA_PATH,
    land_area_data_path=LAND_AREA_DATA_PATH,
    groupby_state=True,
)

vaccines, unimmunized_pop, cases, deaths = solve_folding_horizon(
    solver=solve_nominal_model,
    solver_params={"time_limit": 30, "mip_gap": 0.05, "output_flag": False},
    pop=pop,
    immunized_pop=immunized_pop,
    active_cases=active_cases,
    rep_factor=rep_factor,
    morbidity_rate=morbidity_rate,
    budget=budget,
    planning_horizon=5,
    noise=0.3
)
print(deaths.sum())

