from models.nominal_model import solve_nominal_model
from utils.data_generation import generate_random_data
from models.robust_model import solve_robust_model
from models.robust2_model import solve_robust2_model
from utils.data_processing import get_toy_data_from_census_data
from utils.load_data import process_data
import os


FOLDER_PATH = "/Users/alessandropreviero/Downloads/processed_data"
immunized_pop_data_path = os.path.join(FOLDER_PATH, "immunized_pop.csv")
pop_data_path = os.path.join(FOLDER_PATH, "pop.csv")
active_cases_data_path = os.path.join(FOLDER_PATH, "active_cases.csv")
rep_factor_data_path = os.path.join(FOLDER_PATH, "rep_factor.csv")
morbidity_data_path = os.path.join(FOLDER_PATH, "morbidity_rate.csv")
SOLUTION_PLOT_PATH = "example_plot.png"
TIME_LIMIT = 120
USE_RANDOM_DATA = False

if USE_RANDOM_DATA:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = generate_random_data()

#else:
#    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = get_toy_data_from_census_data(
#        pop_data_path=POP_DATA_PATH,
#        land_area_data_path=LAND_AREA_DATA_PATH,
#        groupby_state=True,
#    )

else:
    states, regions, pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = process_data(
        pop_data_path,
        rep_factor_data_path,
        morbidity_data_path,
        immunized_pop_data_path,
        active_cases_data_path)

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