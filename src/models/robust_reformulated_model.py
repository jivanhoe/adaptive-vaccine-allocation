import logging
from typing import Tuple, Optional
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from src.models.proportional_allocation_model import solve_proportional_allocation_model

logger = logging.getLogger(__name__)


def solve_robust_reformulated_model(
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        budget: np.ndarray,
        min_proportion: float = 0.0,
        sigma: float = 0.2,
        gamma: float = 3.0,
        delta: float = 0.05,
        q_norm: int = 2,
        mip_gap: float = 1e-2,
        feasibility_tol: float = 1e-2,
        output_flag: bool = False,
        time_limit: int = 120
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Initialize model
    m = gp.Model("nominal")

    # ------------------------- PARAMETERS AND VARIABLES ---------------------------------------------------------------

    # Define sets
    num_regions, num_classes = pop.shape
    num_periods = budget.shape[0]
    regions = range(num_regions)
    risk_classes = range(num_classes)
    periods = range(1, num_periods)
    logger.debug(f"Regions: {num_regions} \t Risk classes: {num_classes} \t Periods: {num_periods}")

    # Compute analysis parameters
    compound_rep_factor = np.zeros((num_regions, num_periods))
    cost = np.zeros((num_regions, num_classes, num_periods))
    compound_rep_factor[:, 0] = active_cases.sum(1)
    for t in periods:
        compound_rep_factor[:, t] = compound_rep_factor[:, t - 1] * rep_factor
        for k in risk_classes:
            cost[:, k, t] = morbidity_rate[:, k] * compound_rep_factor[:, t]

    # Compute robustness parameters
    relative_error_lower_bound = np.ones(num_periods)
    relative_error_upper_bound = np.ones(num_periods)
    for t in periods:
        for u in range(t):
            relative_error_lower_bound[t] *= (1 - delta * (np.sqrt(u + 1) - np.sqrt(u))) ** (t - u + 1)
            relative_error_upper_bound[t] *= (1 + delta * (np.sqrt(u + 1) - np.sqrt(u))) ** (t - u + 1)
    rho = (1 + sigma) * np.max(relative_error_upper_bound) - 1
    print(relative_error_upper_bound)

    # Define decision variables
    vaccines = m.addVars(num_regions, num_classes, num_periods)
    unimmunized_pop = m.addVars(num_regions, num_classes, num_periods)
    normalized_cases = m.addVars(num_regions, num_classes, num_periods)  # analysis variable for cases

    # Define analysis variables for robust immunity dynamics
    y = m.addVars(num_regions, num_classes, num_periods)
    l1_penalty = m.addVars(num_regions, num_classes, num_periods)
    lq_dual_penalty = m.addVar()

    # ------------------------- CONSTRAINTS ----------------------------------------------------------------------------

    # Set initial conditions constraints
    m.addConstrs(vaccines[i, k, 0] == 0 for i in regions for k in risk_classes)
    m.addConstrs(normalized_cases[i, k, 0] == 1 for i in regions for k in risk_classes)
    m.addConstrs(unimmunized_pop[i, k, 0] == pop[i, k] - immunized_pop[i, k] for i in regions for k in risk_classes)

    # Set contagion dynamics constraint (bi-linear, non-convex)
    m.addConstrs(
        normalized_cases[i, k, t] == 1 / pop[i].sum() * (
                unimmunized_pop[i, k, t - 1] - vaccines[i, k, t]) * normalized_cases.sum(i, "*", t - 1)
        for i in regions for k in risk_classes for t in periods
    )

    # Set immunity dynamics constraint
    m.addConstrs(
        unimmunized_pop[i, k, t] == unimmunized_pop[i, k, t - 1] - vaccines[i, k, t] -
        relative_error_lower_bound[t] * compound_rep_factor[i, t] * normalized_cases[i, k, t]
        for i in regions for k in risk_classes for t in periods
    )

    # Set budget constraint
    m.addConstrs(vaccines.sum('*', '*', t) <= budget[t] for t in periods)

    # Set general fairness constraint
    m.addConstrs(
        vaccines.sum(i, "*", t) >= min_proportion * unimmunized_pop.sum(i, "*", t)
        for i in regions for t in periods
    )

    # ------------------------- OBJECTIVE ------------------------------------------------------------------------------

    m.addConstrs(
        l1_penalty[i, k, t] >= rho * cost[i, k, t] * normalized_cases[i, k, t] - y[i, k, t]
        for i in regions for k in risk_classes for t in periods
    )
    m.addConstrs(
        l1_penalty[i, k, t] >= - rho * cost[i, k, t] * normalized_cases[i, k, t] + y[i, k, t]
        for i in regions for k in risk_classes for t in periods
    )
    if q_norm == 2:
        m.addConstr(
            lq_dual_penalty * lq_dual_penalty >= sum(
                y[i, k, t] * y[i, k, t] for i in regions for k in risk_classes for t in periods)
        )
    elif q_norm == 1:
        m.addConstrs(
            lq_dual_penalty >= y[i, k, t] for i in regions for k in risk_classes for t in periods
        )
    else:
        raise NotImplementedError

    # Set robust objective
    objective = sum(
        cost[i, k, t] * normalized_cases.sum(i, k, t) + l1_penalty[i, k, t]
        for i in regions for k in risk_classes for t in periods
    ) + gamma * lq_dual_penalty
    m.setObjective(objective, GRB.MINIMIZE)

    # ------------------------- WARM START & MODEL PARAMETERS ----------------------------------------------------------

    vaccines_warm_start, _, _, _ = solve_proportional_allocation_model(
        pop=pop,
        immunized_pop=immunized_pop,
        active_cases=active_cases,
        rep_factor=rep_factor,
        morbidity_rate=morbidity_rate,
        budget=budget,
        delta=delta
    )
    for i in regions:
        for k in risk_classes:
            for t in periods:
                vaccines[i, k, t].start = vaccines_warm_start[i, k, t]

    # Solve model
    m.params.NonConvex = 2
    m.params.MIPGap = mip_gap
    m.params.OutputFlag = output_flag
    m.params.LogToConsole = output_flag
    m.params.FeasibilityTol = feasibility_tol
    m.params.TimeLimit = time_limit
    m.optimize()

    def get_value(variable: gp.tupledict) -> np.array:
        variable = m.getAttr('x', variable)
        return np.array([[[variable[i, k, t] for t in range(num_periods)] for k in risk_classes] for i in regions])

    # Compute solutions
    vaccines = get_value(vaccines)
    unimmunized_pop = get_value(unimmunized_pop)
    normalized_cases = get_value(normalized_cases)
    cases = np.array([
        [
            [compound_rep_factor[i, t] * normalized_cases[i, k, t] for t in range(num_periods)]
            for k in risk_classes
        ] for i in regions
    ])
    deaths = np.array([
        [
            [morbidity_rate[i, k] * (0 if t == 0 else cases[i, k, t - 1]) for t in range(num_periods)]
            for k in risk_classes
        ] for i in regions
    ])

    return vaccines, cases, unimmunized_pop, deaths
