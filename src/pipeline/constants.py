import numpy as np
import datetime as dt

# Input paths
DELPHI_PARAMS_PATH = "../../data/inputs/delphi-parameters.csv"
DELPHI_PREDICTIONS_PATH = "../../data/inputs/delphi-predictions.csv"
CDC_DATA_PATH = "../../data/inputs/cdc-data.csv"
POPULATION_DATA_PATH = "../../data/inputs/us-population.csv"

# Output paths
MODEL_PATH_PATH = "../../data/outputs/model-"
MORTALITY_RATES_PATH = "../../data/outputs/mortality-rates-"
BASELINE_SOLUTION_PATH = "../../data/outputs/baseline-solution-"
OPTIMIZED_SOLUTION_ATH = "../../data/outputs/optimized-solution-"
RESULTS_PATH = "../../data/outputs/results.csv"

# Population partition
RISK_CLASSES = [
    dict(min_age=0.0, max_age=9.0),
    dict(min_age=10.0, max_age=49.0),
    dict(min_age=50.0, max_age=59.0),
    dict(min_age=60.0, max_age=69.0),
    dict(min_age=70.0, max_age=79.0),
    dict(min_age=80.0, max_age=np.inf)
]
N_REGIONS = 51  # All 50 US states plus Washington D.C.
N_RISK_CLASSES = len(RISK_CLASSES)

# Time discretization
DAYS_PER_TIMESTEP = 1.0

# Mortality rate estimation parameters
RESCALE_BASELINE = True
N_TIMESTEPS_PER_ESTIMATE = 10
MIN_LAG = 2
MAX_LAG = 20
MAX_PCT_CHANGE = 0.2
MIN_MORTALITY_RATE = 0
MAX_MORTALITY_RATE = 0.2
REGULARIZATION_PARAM = 0.1

# Coordinate descent algorithm parameters
EXPLORATION_TOL = 5e2
TERMINATION_TOL = 5e2
MAX_ITERATIONS = 10
N_EARLY_STOPPING_ITERATIONS = 1

# Gurobi parameters
TIME_LIMIT = 200
FEASIBILITY_TOL = 1e-3
MIP_GAP = 5e-2
BARRIER_CONV_TOL = 1e-5

# Fixed DELPHI parameters
DETECTION_PROBABILITY = 0.2
MEDIAN_PROGRESSION_TIME = 5.0
MEDIAN_DETECTION_TIME = 2.0
MEDIAN_HOSPITALIZED_RECOVERY_TIME = 10.0
MEDIAN_UNHOSPITALIZED_RECOVERY_TIME = 15.0

# Vaccine parameters
MAX_ALLOCATION_FACTOR = 10
MIN_ALLOCATION_FACTOR = 0.1
MAX_DECREASE_PCT = 0.1
MAX_INCREASE_PCT = 0.1
MAX_TOTAL_CAPACITY_PCT = None
OPTIMIZE_CAPACITY = False
EXCLUDED_RISK_CLASSES = [0, 5]

# Scenarios
DATES_GRID = [
    dict(start_date=dt.datetime(2020, 4, 20), end_date=dt.datetime(2020, 7, 15)),
    dict(start_date=dt.datetime(2020, 7, 15), end_date=dt.datetime(2020, 10, 15))
]
VACCINE_EFFECTIVENESS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
DAILY_VACCINE_BUDGET_GRID = [1e5, 2e5, 3e5, 5e5, 1e6, 2e6]
MIN_ALLOCATION_FACTOR_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

