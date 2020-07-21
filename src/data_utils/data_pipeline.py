import datetime as dt

import pandas as pd

from data_utils.constants import *
from data_utils.data_loading import load_and_clean_delphi_params, load_and_clean_delphi_predictions
from data_utils.data_processing import get_initial_conditions, get_delphi_params, get_vaccine_params
from models.prescriptive_delphi_model import PrescriptiveDELPHIModel


def load_model(
        start_date: dt.datetime,
        end_date: dt.datetime
) -> PrescriptiveDELPHIModel:

    # Load raw data
    params_df = load_and_clean_delphi_params(DELPHI_PARAMS_PATH)
    predictions_df = load_and_clean_delphi_predictions(DELPHI_PREDICTIONS_PATH)
    cdc_df = pd.read_csv(CDC_DATA_PATH)
    pop_df = pd.read_csv(POPULATION_DATA_PATH)

    # Get processed data for model
    initial_conditions = get_initial_conditions(
        pop_df=pop_df,
        predictions_df=predictions_df,
        start_date=start_date
    )
    delphi_params = get_delphi_params(
        pop_df=pop_df,
        cdc_df=cdc_df,
        params_df=params_df,
        predictions_df=predictions_df,
        start_date=start_date,
        end_date=end_date
    )
    vaccine_params = get_vaccine_params(
        total_pop=initial_conditions["population"].sum(),
        start_date=start_date,
        end_date=end_date
    )

    # Return prescriptive DELPHI model object
    return PrescriptiveDELPHIModel(
        initial_conditions=initial_conditions,
        delphi_params=delphi_params,
        vaccine_params=vaccine_params
    )
