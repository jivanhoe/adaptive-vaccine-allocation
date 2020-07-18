import numpy as np

# Paths
DELPHI_PARAMS_PATH = "../data/delphi-parameters.csv"
DELPHI_PREDICTIONS_PATH = "../data/delphi-predictions.csv"
CDC_DATA_PATH = "../data/cdc-data.csv"
POPULATION_DATA_PATH = "../data/us-population.csv"

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

# Planning horizon and granularity
N_TIMESTEPS = 90  # 3-month planning period
DAYS_PER_TIMESTEP = 1.0

# Mortality rate estimation parameters
MAX_PCT_CHANGE = 0.2
MAX_PCT_POPULATION_DEVIATION = 0.3
N_TIMESTEPS_PER_ESTIMATE = 10

# Fixed DELPHI parameters
DETECTION_PROBABILITY = 0.2
MEDIAN_PROGRESSION_TIME = 5.0
MEDIAN_DETECTION_TIME = 2.0
MEDIAN_HOSPITALIZED_DEATH_TIME = 20.0
MEDIAN_UNHOSPITALIZED_DEATH_TIME = 15.0
MEDIAN_HOSPITALIZED_RECOVERY_TIME = 10.0
MEDIAN_UNHOSPITALIZED_RECOVERY_TIME = 15.0
