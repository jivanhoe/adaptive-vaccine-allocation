from models.nominal_model import solve_nominal_model
from utils.data_generation import generate_random_data
from models.robust_model import solve_robust_model
from models. robust_reformulated_model import solve_robust_reformulated_model
from utils.data_processing import load_data

DATA_DIR = "../../data/processed/"
USE_RANDOM_DATA = False

if USE_RANDOM_DATA:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = generate_random_data()
else:
    pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = load_data(data_dir=DATA_DIR)
    rep_factor = rep_factor[:, 0]
print(pop.shape, immunized_pop.shape, active_cases.shape, rep_factor.shape, morbidity_rate.shape, budget.shape)

vaccines, cases, unimmunized_pop, deaths = solve_robust_model(
    pop=pop,
    immunized_pop=immunized_pop,
    active_cases=active_cases,
    rep_factor=rep_factor,
    morbidity_rate=morbidity_rate,
    budget=budget[:5],
    time_limit=120,
    output_flag=True
)


