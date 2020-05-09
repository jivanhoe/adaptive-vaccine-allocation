import logging
from typing import Tuple, Optional
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from src.models.proportional_allocation_model import solve_proportional_allocation_model

logger = logging.getLogger(__name__)


def compute_uncertain_factors(
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        active_cases: np.ndarray,
        num_regions: int,
        num_classes: int,
        num_periods: int
) -> Tuple[np.ndarray, np.ndarray]:
    var_rep_fac = np.zeros(shape=(num_regions, num_periods))
    var_morbidity = np.zeros(shape=(num_regions, num_classes, num_periods))

    for r in range(num_regions):
        for t in range(1, num_periods):
            var_rep_fac[r, t] = (rep_factor[r] ** t) * sum(active_cases[r, c] for c in range(num_classes))

            for c in range(num_classes):
                var_morbidity[r, c, t] = morbidity_rate[r, c] * var_rep_fac[r, t]

    return var_rep_fac, var_morbidity


def solve_robust2_model(
        pop: np.ndarray,
        immunized_pop: np.ndarray,
        active_cases: np.ndarray,
        rep_factor: np.ndarray,
        morbidity_rate: np.ndarray,
        budget: np.ndarray,
        min_proportion: float = 0.0,
        rho: float = 0.1,
        gamma: Optional[float] = None,
        delta: float = 0.1,
        q_norm: int = 2,
        mip_gap: float = 1e-4,
        feasibility_tol: float = 1e-4,
        output_flag: bool = True,
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

    # Compute new uncertain parameters
    var_rep_fac, var_morbidity = compute_uncertain_factors(rep_factor, morbidity_rate, active_cases,
                                                           num_regions, num_classes, num_periods)

    # Define decision variables
    vaccines = m.addVars(num_regions, num_classes, num_periods)
    var_cases = m.addVars(num_regions, num_classes, num_periods)  # == cases / var_rep_fac
    unimmunized_pop = m.addVars(num_regions, num_classes, num_periods)

    # Define analysis variables for robust objective
    l1_morbidity = m.addVars(num_regions, num_classes, num_periods)
    lq_morbidity = m.addVar()

    # Set gamma to default values (for 99% probabilistic guarantee) if not specified
    if gamma is None:
        if q_norm == 2:
            gamma = 3.03 * rho ** 2
        elif q_norm == 1:
            gamma = 3.03 * np.sqrt(num_regions * num_classes * num_periods) * rho ** 2
        else:
            raise NotImplementedError

    # ------------------------- CONSTRAINTS ----------------------------------------------------------------------------

    # Set initial conditions constraints
    m.addConstrs(vaccines[i, k, 0] == 0 for i in regions for k in risk_classes)
    m.addConstrs(var_cases[i, k, 0] == 1 for i in regions for k in risk_classes)
    m.addConstrs(unimmunized_pop[i, k, 0] == pop[i, k] - immunized_pop[i, k] for i in regions for k in risk_classes)

    # Set robust immunity dynamics constraints
    m.addConstrs(
        unimmunized_pop[i, k, t] == unimmunized_pop[i, k, t - 1] - vaccines[i, k, t] -
        (1-rho) * var_rep_fac[i, t] * var_cases[i, k, t] for i in regions for k in risk_classes for t in periods
    )

    # Set contagion dynamics (bi-linear constraints)
    m.addConstrs(
        var_cases[i, k, t] == 1 / pop[i].sum() * (unimmunized_pop[i, k, t - 1] - vaccines[i, k, t]) *
        var_cases.sum(i, "*", t - 1) for i in regions for k in risk_classes for t in periods
    )

    # Set budget constraint
    m.addConstrs(vaccines.sum('*', '*', t) <= budget[t] for t in periods)

    # Set general fairness constraint
    m.addConstrs(
        vaccines.sum(i, "*", t) >= min_proportion * unimmunized_pop.sum(i, "*", t)
        for i in regions for t in periods
    )

    # ------------------------- OBJECTIVE ------------------------------------------------------------------------------

    m.addConstrs(l1_morbidity[i, k, t] >= var_morbidity[i, k, t] * var_cases[i, k, t]
                 for i in regions for k in risk_classes for t in periods
                 )

    m.addConstrs(l1_morbidity[i, k, t] >= -var_morbidity[i, k, t] * var_cases[i, k, t]
                 for i in regions for k in risk_classes for t in periods
                 )

    if q_norm == 2:
        m.addConstr(
            lq_morbidity * lq_morbidity >=
            sum((var_morbidity[i, k, t] * var_cases[i, k, t]) * (var_morbidity[i, k, t] * var_cases[i, k, t])
                for i in regions for k in risk_classes for t in periods)
        )

    elif q_norm == 1:
        m.addConstrs(
            lq_morbidity >= var_morbidity[i, k, t] * var_cases[i, k, t]
            for i in regions for k in risk_classes for t in periods
        )
    else:
        raise NotImplementedError

    # Set robust objective
    objective = sum(var_morbidity[i, k, t] * var_cases[i, k, t] +
                    rho * l1_morbidity[i, k, t] for i in regions for k in risk_classes for t in periods) + \
                gamma * lq_morbidity

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
    var_cases = get_value(var_cases)
    cases = np.array([
        [
            [var_cases[i, k, t]*var_rep_fac[i, t] for t in range(num_periods)]
            for k in risk_classes
        ] for i in regions
    ])

    unimmunized_pop = get_value(unimmunized_pop)
    deaths = np.array([
        [
            [morbidity_rate[i, k] * (0 if t == 0 else cases[i, k, t - 1]) for t in range(num_periods)]
            for k in risk_classes
        ] for i in regions
    ])

    return vaccines, cases, unimmunized_pop, deaths
