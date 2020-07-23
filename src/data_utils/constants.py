import numpy as np

# Paths
DELPHI_PARAMS_PATH = "../../data/inputs/delphi-parameters.csv"
DELPHI_PREDICTIONS_PATH = "../../data/inputs/delphi-predictions.csv"
CDC_DATA_PATH = "../../data/inputs/cdc-data.csv"
POPULATION_DATA_PATH = "../../data/inputs/us-population.csv"

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
N_TIMESTEPS_PER_ESTIMATE = 5
MAX_PCT_CHANGE = 0.2
MIN_MORTALITY_RATE = 5e-4
MAX_MORTALITY_RATE = 0.5
REGULARIZATION_PARAM = 1e-3  # Only used if cases deviation constraint is relaxed
USE_L2_ERROR = True
TIME_LIMIT = 60
FEASIBILITY_TOL = 1e-3
MIP_GAP = 1e-2

# Fixed DELPHI parameters
DETECTION_PROBABILITY = 0.2
MEDIAN_PROGRESSION_TIME = 5.0
MEDIAN_DETECTION_TIME = 2.0
MEDIAN_HOSPITALIZED_RECOVERY_TIME = 10.0
MEDIAN_UNHOSPITALIZED_RECOVERY_TIME = 15.0

# Vaccine parameters
VACCINE_EFFECTIVENESS = 0.6
VACCINE_BUDGET_PCT = 5e-4
MAX_ALLOCATION_PCT = 5e-3
MIN_ALLOCATION_PCT = 5e-5
MAX_DECREASE = 0.1
MAX_INCREASE = 0.1
MAX_TOTAL_CAPACITY_PCT = None
OPTIMIZE_CAPACITY = False
EXCLUDED_RISK_CLASSES = [0, 5]
