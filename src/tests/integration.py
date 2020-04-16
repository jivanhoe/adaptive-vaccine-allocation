from models.nominal_model import solve_nominal_model
from utils.data_generation import generate_random_data

pop, immunized_pop, active_cases, rep_factor, morbidity_rate, budget = generate_random_data()

allocated_vaccines = solve_nominal_model(
    pop=pop,
    immunized_pop=immunized_pop,
    active_cases=active_cases,
    rep_factor=rep_factor,
    morbidity_rate=morbidity_rate,
    budget=budget,
    time_limit=10
)

