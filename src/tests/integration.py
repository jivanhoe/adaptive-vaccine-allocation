from src.models.nominal_model import solve_nominal_model
from src.utils.data_generation import generate_random_data
from src.utils.data_processing import get_toy_data_from_census_data
from src.utils.solution_visualization import plot_solution

POP_DATA_PATH = "../../data/census/pop-data.csv"
LAND_AREA_DATA_PATH = "../../data/census/land-area-data.csv"
SOLUTION_PLOT_PATH = "example_plot.png"
TIME_LIMIT = 120
USE_RANDOM_DATA = False
SAVE_PLOT = True

if USE_RANDOM_DATA:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = generate_random_data()
else:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
        pop_data_path=POP_DATA_PATH,
        land_area_data_path=LAND_AREA_DATA_PATH,
        groupby_state=True,
    )

vaccines, cases, unimmunized_pop, deaths = solve_nominal_model(
    pop=pop,
    immunized_pop=immunized_pop,
    active_cases=active_cases,
    rep_factor=rep_factor,
    morbidity_rate=morbidity_rate,
    budget=budget[:7],
    time_limit=TIME_LIMIT
)

if SAVE_PLOT:
    plot_solution(
        vaccines=vaccines,
        cases=cases,
        unimmunized_pop=unimmunized_pop,
        deaths=deaths,
        path=SOLUTION_PLOT_PATH
    )
