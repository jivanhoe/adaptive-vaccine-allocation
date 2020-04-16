import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_nominal_model(
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        budget: np.ndarray,
        alpha: float = 0.5,
        time_limit: int = 60,
        mip_gap: int = 1e-2
):

    # Initialize model
    m = gp.Model("nominal")

    # Define sets
    num_regions = pop.shape[0]
    num_classes = pop.shape[1]
    num_periods = budget.shape[0]
    regions = range(num_regions)
    risk_classes = range(num_classes)
    periods = range(1, num_periods)

    # Define decision variables
    vaccines = m.addVars(num_regions, num_classes, num_periods, lb=0)
    cases = m.addVars(num_regions, num_classes, num_periods, lb=0)
    unimmunized_pop = m.addVars(num_regions, num_classes, num_periods, lb=0)

    # Set initial conditions constraints
    m.addConstrs(vaccines[i, k, 0] == 0 for i in regions for k in risk_classes)
    m.addConstrs(cases[i, k, 0] == active_cases[i, k] for i in regions for k in risk_classes)
    m.addConstrs(
        unimmunized_pop[i, k, 0] == pop[i, k] - immunized_pop[i, k]
        for i in regions for k in risk_classes
    )

    # Set immunity dynamics constraint
    m.addConstrs(
        unimmunized_pop[i, k, t] >= unimmunized_pop[i, k, t - 1] - cases[i, k, t - 1] - vaccines[i, k, t]
        for i in regions for k in risk_classes for t in periods
    )

    # Set contagion dynamics constraint (bi-linear, non-convex)
    m.addConstrs(
        cases[i, k, t] >= rep_factor[i] / pop[i].sum() * unimmunized_pop[i, k, t] * cases.sum(i, "*", t-1)
        for i in regions for k in risk_classes for t in periods
    )

    # Set budget constraint
    m.addConstrs(vaccines.sum('*', '*', t) <= budget[t] for t in periods)

    # Set general fairness constraint
    m.addConstrs(
        vaccines.sum(i, "*", t) >= alpha * pop[i].sum() / pop.sum() * budget[t]
        for i in regions for t in periods
    )

    # Define objective
    objective = sum(morbidity_rate[k] * cases.sum("*", k, "*") for k in risk_classes)
    m.setObjective(objective, GRB.MINIMIZE)

    # Solve model
    m.params.NonConvex = 2
    m.params.TimeLimit = time_limit
    m.params.MIPGap = mip_gap
    m.optimize()

    # Return allocated vaccines
    allocated_vaccines = m.getAttr('x', vaccines)
    return np.array([[[allocated_vaccines[i, k, t] for t in periods] for k in risk_classes] for i in regions])
