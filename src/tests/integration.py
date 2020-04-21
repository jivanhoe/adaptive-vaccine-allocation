from src.models.nominal_model import solve_nominal_model
from src.models.proportional_allocation_model import solve_proportional_allocation_model
from src.models.mc_convex_relaxation import solve_linear_relaxation
from src.utils.data_generation import generate_random_data
from src.utils.data_processing import get_toy_data_from_census_data
from src.utils.solution_visualization import plot_solution

POP_DATA_PATH = "../../data/census/pop-data.csv"
LAND_AREA_DATA_PATH = "../../data/census/land-area-data.csv"
SOLUTION_PLOT_PATH = "example_plot.png"
TIME_LIMIT = 10
USE_RANDOM_DATA = True
USE_PROPORTIONAL_ALLOCATION_MODEL = False
USE_LINEAR_RELAXATION_MODEL = False
SAVE_PLOT = True

if USE_RANDOM_DATA:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = generate_random_data()
else:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
        pop_data_path=POP_DATA_PATH,
        land_area_data_path=LAND_AREA_DATA_PATH,
        groupby_state=True
    )

if USE_PROPORTIONAL_ALLOCATION_MODEL:
    vaccines, cases, unimmunized_pop, deaths = solve_proportional_allocation_model(
        pop=pop,
        immunized_pop=immunized_pop,
        active_cases=active_cases,
        rep_factor=rep_factor,
        morbidity_rate=morbidity_rate,
        budget=budget
    )
    print(f"Objective value: {deaths.sum()}")

elif USE_LINEAR_RELAXATION_MODEL:
    vaccines, cases, unimmunized_pop, deaths = solve_linear_relaxation(
        pop=pop,
        immunized_pop=immunized_pop,
        active_cases=active_cases,
        rep_factor=rep_factor,
        morbidity_rate=morbidity_rate,
        budget=budget
    )
    print(f"Objective value: {deaths.sum()}")

else:
    vaccines, cases, unimmunized_pop, deaths = solve_nominal_model(
        pop=pop,
        immunized_pop=immunized_pop,
        active_cases=active_cases,
        rep_factor=rep_factor,
        morbidity_rate=morbidity_rate,
        budget=budget,
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
