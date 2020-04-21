import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_linear_relaxation(
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        budget: np.ndarray,
        alpha: float = 0.5):

    # Define model
    m = gp.Model("relaxation")

    # Define sets
    num_regions = pop.shape[0]
    num_classes = pop.shape[1]
    num_periods = budget.shape[0]
    regions = range(num_regions)
    risk_classes = range(num_classes)
    periods = range(1, num_periods)

    # Define Upper and Lower Bounds for relaxation
    LB_part_un = np.zeros(shape=(num_regions, num_classes, num_periods))
    LB_aggr_cas = np.zeros(shape=(num_regions, num_periods))
    UB_part_un = np.repeat(pop[:, :, np.newaxis], num_periods, axis=2)
    aggregate_pop = np.sum(pop, 1)
    UB_aggr_cas = np.repeat(aggregate_pop[:, np.newaxis], num_periods, axis=1)

    # Define variables (default is lb = 0, ub = inf, var_type = GRB.CONTINUOUS)
    vaccines = m.addVars(num_regions, num_classes, num_periods)
    cases = m.addVars(num_regions, num_classes, num_periods)
    unimmunized_pop = m.addVars(num_regions, num_classes, num_periods)
    aggregate_cases = m.addVars(num_regions, num_periods, lb=LB_aggr_cas, ub=UB_aggr_cas)
    partial_unim = m.addVars(num_regions, num_classes, num_periods, lb=LB_part_un, ub=UB_part_un)

    # Set initial conditions (at time = 0)
    m.addConstrs(vaccines[i, k, 0] == 0 for i in regions for k in risk_classes)
    m.addConstrs(cases[i, k, 0] == active_cases[i, k] for i in regions for k in risk_classes)
    m.addConstrs(unimmunized_pop[i, k, 0] == pop[i, k] - immunized_pop[i, k] for i in regions for k in risk_classes)

    m.addConstrs(aggregate_cases[i, 0] == active_cases[i].sum() for i in regions)
    m.addConstrs(partial_unim[i, k, 0] == pop[i, k] - immunized_pop[i, k] for i in regions for k in risk_classes)

    # Set immunity dynamics constraints
    m.addConstrs(
        unimmunized_pop[i, k, t] == unimmunized_pop[i, k, t - 1] - vaccines[i, k, t] - cases[i, k, t]
        for i in regions for k in risk_classes for t in periods
    )

    # Linearise bilinear constraint of contagion dynamics
    m.addConstrs(aggregate_cases[i, t] == cases.sum(i, '*', t) for i in regions for t in periods)
    m.addConstrs(
        partial_unim[i, k, t] == rep_factor[i] / pop[i].sum() * (unimmunized_pop[i, k, t - 1] - vaccines[i, k, t])
        for i in regions for k in risk_classes for t in periods)

    # Add general McCormick Envelopes
    m.addConstrs(cases[i, k, t] >=
                 LB_part_un[i, k, t] * aggregate_cases[i, t - 1] +
                 partial_unim[i, k, t] * LB_aggr_cas[i, t - 1] -
                 LB_part_un[i, k, t] * LB_aggr_cas[i, t - 1]
                 for i in regions for k in risk_classes for t in periods)

    m.addConstrs(cases[i, k, t] >=
                 UB_part_un[i, k, t] * aggregate_cases[i, t - 1] +
                 partial_unim[i, k, t] * UB_aggr_cas[i, t - 1] -
                 UB_part_un[i, k, t] * UB_aggr_cas[i, t - 1]
                 for i in regions for k in risk_classes for t in periods)

    m.addConstrs(cases[i, k, t] <=
                 UB_part_un[i, k, t] * aggregate_cases[i, t - 1] +
                 partial_unim[i, k, t] * LB_aggr_cas[i, t - 1] -
                 UB_part_un[i, k, t] * LB_aggr_cas[i, t - 1]
                 for i in regions for k in risk_classes for t in periods)

    m.addConstrs(cases[i, k, t] <=
                 partial_unim[i, k, t] * UB_aggr_cas[i, t - 1] +
                 LB_part_un[i, k, t] * aggregate_cases[i, t - 1] -
                 LB_part_un[i, k, t] * UB_aggr_cas[i, t - 1]
                 for i in regions for k in risk_classes for t in periods)

    # Set budget constraint
    m.addConstrs(vaccines.sum('*', '*', t) <= budget[t] for t in periods)

    # Set general fairness constraint
    m.addConstrs(
        vaccines.sum(i, "*", t) >= alpha * pop[i].sum() / pop.sum() * budget[t]
        for i in regions for t in periods
    )

    # Set Objective & solve model
    objective = sum(morbidity_rate[k] * cases.sum("*", k, "*") for k in risk_classes)
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
            [morbidity_rate[k] * (0 if t == 0 else cases[i, k, t - 1]) for t in range(num_periods)]
            for k in risk_classes
        ] for i in regions
    ])
    return vaccines, cases, unimmunized_pop, deaths
