import numpy as np
import gurobipy as gp
from gurobipy import GRB

from typing import Tuple

from models.proportional_allocation_model import solve_proportional_allocation_model


def solve_convex_relaxation(
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        budget: np.ndarray,
        alpha: float = 0.5
):

    # Define model
    m = gp.Model("relaxation")

    # Define sets
    num_regions = pop.shape[0]
    num_classes = pop.shape[1]
    num_periods = budget.shape[0]
    regions = range(num_regions)
    risk_classes = range(num_classes)
    periods = range(1, num_periods)

    # Set upper and lower bounds for aggregate cases
    _, max_cases, _, _ = solve_proportional_allocation_model(
        pop=pop,
        immunized_pop=immunized_pop,
        active_cases=active_cases,
        rep_factor=rep_factor,
        budget=np.zeros(num_periods),
        morbidity_rate=morbidity_rate
    )
    aggregate_cases_lb = np.zeros(shape=(num_regions, num_periods))
    aggregate_cases_ub = np.zeros(shape=(num_regions, num_periods))
    aggregate_cases_lb[:, 0] = active_cases.sum(1)
    aggregate_cases_ub[:, 0] = active_cases.sum(1)
    for i in regions:
        aggregate_cases_ub[i, 1:] = max_cases[i, :].sum(0).max()

    # Set upper and lower bounds for cases for infections per case envelopes
    infections_per_case_lb = np.zeros(shape=(num_regions, num_classes, num_periods))
    infections_per_case_ub = np.zeros(shape=(num_regions, num_classes, num_periods))
    for i in regions:
        for k in risk_classes:
            infections_per_case_ub[i, k, :] = 2 * rep_factor[i] / pop[i].sum() * (pop[i, k] - immunized_pop[i, k])

    # Define variables
    vaccines = m.addVars(num_regions, num_classes, num_periods, lb=0)
    cases = m.addVars(num_regions, num_classes, num_periods, lb=0)
    unimmunized_pop = m.addVars(num_regions, num_classes, num_periods, lb=0)
    aggregate_cases = m.addVars(num_regions, num_periods, lb=aggregate_cases_lb, ub=aggregate_cases_ub)
    infections_per_case = m.addVars(num_regions, num_classes, num_periods, lb=infections_per_case_lb, ub=infections_per_case_ub)

    # Set initial conditions (at time = 0)
    m.addConstrs(vaccines[i, k, 0] == 0 for i in regions for k in risk_classes)
    m.addConstrs(cases[i, k, 0] == active_cases[i, k] for i in regions for k in risk_classes)
    m.addConstrs(unimmunized_pop[i, k, 0] == pop[i, k] - immunized_pop[i, k] for i in regions for k in risk_classes)
    m.addConstrs(aggregate_cases[i, 0] == active_cases[i].sum() for i in regions)
    m.addConstrs(infections_per_case[i, k, 0] == pop[i, k] - immunized_pop[i, k] for i in regions for k in risk_classes)

    # Set immunity dynamics constraints
    m.addConstrs(
        unimmunized_pop[i, k, t] >= unimmunized_pop[i, k, t - 1] - vaccines[i, k, t] - cases[i, k, t]
        for i in regions for k in risk_classes for t in periods
    )

    # Linearize bi-linear constraints for contagion dynamics using McCormick envelopes
    m.addConstrs(aggregate_cases[i, t] == cases.sum(i, '*', t) for i in regions for t in periods)
    m.addConstrs(
        infections_per_case[i, k, t] >= rep_factor[i] / pop[i].sum() * (unimmunized_pop[i, k, t - 1] - vaccines[i, k, t])
        for i in regions for k in risk_classes for t in periods
    )
    m.addConstrs(
        cases[i, k, t] >=
        infections_per_case_lb[i, k, t] * aggregate_cases[i, t - 1] +
        infections_per_case[i, k, t] * aggregate_cases_lb[i, t - 1] -
        infections_per_case_lb[i, k, t] * aggregate_cases_lb[i, t - 1]
        for i in regions for k in risk_classes for t in periods
    )
    m.addConstrs(
        cases[i, k, t] >=
        infections_per_case_ub[i, k, t] * aggregate_cases[i, t - 1] +
        infections_per_case[i, k, t] * aggregate_cases_ub[i, t - 1] -
        infections_per_case_ub[i, k, t] * aggregate_cases_ub[i, t - 1]
        for i in regions for k in risk_classes for t in periods
    )
    m.addConstrs(
        cases[i, k, t] <=
        infections_per_case_ub[i, k, t] * aggregate_cases[i, t - 1] +
        infections_per_case[i, k, t] * aggregate_cases_lb[i, t - 1] -
        infections_per_case_ub[i, k, t] * aggregate_cases_lb[i, t - 1]
        for i in regions for k in risk_classes for t in periods
    )
    m.addConstrs(
        cases[i, k, t] <=
        infections_per_case_lb[i, k, t] * aggregate_cases[i, t - 1] +
        infections_per_case[i, k, t] * aggregate_cases_ub[i, t - 1] -
        infections_per_case_lb[i, k, t] * aggregate_cases_ub[i, t - 1]
        for i in regions for k in risk_classes for t in periods
    )

    # Set budget constraint
    m.addConstrs(vaccines.sum('*', '*', t) <= budget[t] for t in periods)

    # Set general fairness constraint
    m.addConstrs(
        vaccines.sum(i, "*", t) >= alpha * pop[i].sum() / pop.sum() * budget[t]
        for i in regions for t in periods
    )

    # Set Objective & solve model
    objective = sum(morbidity_rate[i, k] * cases.sum("*", k, "*") for i in regions for k in risk_classes)
    m.setObjective(objective, GRB.MINIMIZE)

    m.optimize()

    # Return optimal variables
    def get_value(variable: gp.tupledict) -> np.array:
        variable = m.getAttr('x', variable)
        return np.array([[[variable[i, k, t] for t in range(num_periods)] for k in risk_classes] for i in regions])
    vaccines = get_value(vaccines)
    cases = get_value(cases)
    unimmunized_pop = get_value(unimmunized_pop)
    deaths = np.array([
        [
            [morbidity_rate[i, k] * (0 if t == 0 else cases[i, k, t - 1]) for t in range(num_periods)]
            for k in risk_classes
        ] for i in regions
    ])
    return vaccines, cases, unimmunized_pop, deaths
