from models.nominal_model import solve_nominal_model
from utils.data_generation import generate_random_data
from utils.data_processing import get_toy_data_from_census_data

POP_DATA_PATH = "../../data/census/pop-data.csv"
LAND_AREA_DATA_PATH = "../../data/census/land-area-data.csv"
TIME_LIMIT = 200
USE_RANDOM_DATA = False

if USE_RANDOM_DATA:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = generate_random_data()
else:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
        pop_data_path=POP_DATA_PATH,
        land_area_data_path=LAND_AREA_DATA_PATH,
        groupby_state=True
    )

allocated_vaccines = solve_nominal_model(
    pop=pop,
    immunized_pop=immunized_pop,
    active_cases=active_cases,
    rep_factor=rep_factor,
    morbidity_rate=morbidity_rate,
    budget=budget,
    time_limit=TIME_LIMIT
)

