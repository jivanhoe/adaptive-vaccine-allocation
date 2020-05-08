from models.nominal_model import solve_nominal_model
from utils.data_generation import generate_random_data
from models.robust_model import solve_robust_model
from models.robust2_model import solve_robust2_model
from utils.data_processing import get_toy_data_from_census_data

POP_DATA_PATH = "../../data/census/pop-data.csv"
LAND_AREA_DATA_PATH = "../../data/census/land-area-data.csv"
SOLUTION_PLOT_PATH = "example_plot.png"
TIME_LIMIT = 120
USE_RANDOM_DATA = False

if USE_RANDOM_DATA:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = generate_random_data()
else:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
        pop_data_path=POP_DATA_PATH,
        land_area_data_path=LAND_AREA_DATA_PATH,
        groupby_state=True,
    )

vaccines, cases, unimmunized_pop, deaths = solve_robust_model(
    pop=pop,
    immunized_pop=immunized_pop,
    active_cases=active_cases,
    rep_factor=rep_factor,
    morbidity_rate=morbidity_rate,
    budget=budget[:5],
    time_limit=TIME_LIMIT,
    mip_gap=1e-4,
    delta=0.05,
    rho=0.01
)


