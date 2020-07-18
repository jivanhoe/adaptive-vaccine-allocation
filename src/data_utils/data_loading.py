import pandas as pd


def load_and_clean_delphi_params(path: str) -> pd.DataFrame:
    # Load data from CSV file
    df = pd.read_csv(path)

    # Filter down to US only
    df = df[
        df["Country"] == "US"
        ].drop(["Continent", "Country", "MAPE"], axis=1)

    # Rename columns
    df.rename(
        columns={
            "Province": "state",
            "Data Start Date": "start_date",
            "Median Day of Action": "intervention_time",
            "Rate of Action": "intervention_rate",
            "Infection Rate": "infection_rate",
            "Rate of Death": "death_rate",
            "Mortality Rate": "mortality_rate",
            "Rate of Mortality Rate Decay": "mortality_rate_decay",
            "Internal Parameter 1": "exposed_initial_param",
            "Internal Parameter 2": "infected_initial_param",
            "Jump Magnitude": "jump_magnitude",
            "Jump Time": "jump_time",
            "Jump Decay": "jump_decay"
        },
        inplace=True
    )

    # Cast start date as datetime object
    df["start_date"] = pd.to_datetime(df["start_date"])

    return df.set_index("state")


def load_and_clean_delphi_predictions(path: str) -> pd.DataFrame:
    # Load data from CSV file
    df = pd.read_csv(path)

    # Filter down to US only
    df = df[df["Country"] == "US"]

    # Aggregate intermediary state
    df["recovering"] = df[["AR", "DHR", "DQR"]].sum(1)
    df["dying"] = df[["AD", "DHD", "DQD"]].sum(1)

    # Select relevant columns and rename
    df = df[
        [
            "Province",
            "Day",
            "S",
            "E",
            "I",
            "R",
            "D",
            "recovering",
            "dying"
        ]
    ].rename(
        columns={
            "Province": "state",
            "Day": "date",
            "S": "susceptible",
            "E": "exposed",
            "I": "infectious",
            "R": "recovered",
            "D": "deceased",
        }
    )

    # Cast date column as datetime object
    df["date"] = pd.to_datetime(df["date"])

    return df
