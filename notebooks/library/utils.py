import os

import lightgbm as lgb
import numpy as np
import pandas as pd

data_path = "../data/"


# boundaries_sub_data_path = "other/boundaries"
# bayesian_run_path = "../data/bayesian_runs/"

def read_test(df_train, df_census_population):
    df_test = read_df("test.csv")
    t1 = df_train[["cfips", "state", "state_abb", "county"]].value_counts().reset_index().drop(columns=[0])
    df_test = pd.merge(
        df_test, t1, "left", "cfips"
    )
    df_test = _fix_df_train(df_test)

    # Remove dates already published by Kaggle
    df_test = df_test[~(df_test['first_day_of_month'].isin(['2022-11-01', '2022-12-01']))]

    # Join population data
    df_test = pd.merge(df_test, df_census_population, 'left', ['year', 'county', 'state', 'state_abb'])

    df_test = df_test.rename(columns={'census_over_18_population_x1000': 'target_census_over_18_population_x1000',
                                      'census_population_x1000': 'target_census_population_x1000'})

    return df_test


def read_df(filename, sub_folder="kaggle", delimiter=",", skiprows=0):
    df = pd.read_csv(os.path.join(data_path, sub_folder, filename), delimiter=delimiter, skiprows=skiprows)

    # if filename == "train.csv" or filename == 'test.csv':
    if filename == "train.csv":
        df_revealed_test = read_df("revealed_test.csv")
        df = (
            pd.concat([df, df_revealed_test], axis=0)
            .sort_values(["cfips", "first_day_of_month"])
            .reset_index(drop=True)
        )
        df['state_abb'] = df.apply(lambda x: states[x['state']], axis=1)
        df = _fix_df_train_issues(df)
        df = _fix_df_train(df)
    return df


def write_df(df, file_name, subfolder):
    path = os.path.join(data_path, subfolder)
    df.to_csv(os.path.join(path, file_name), index=False)


def smape(pred, eval_data):
    if hasattr(eval_data, "label"):
        A = eval_data.label  # Used by lightgbm
    else:
        A = eval_data  # Used by numpy
    F = pred

    if type(pred) == int or type(pred) == float:
        # Single cases
        value = 100 / 1 * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    else:
        # Many cases
        value = 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    return "smape", value, False


def build_callbacks(
        early_stopping: int = 0, log_evaluation: int = 0, record_evaluation: dict = None
):
    callbacks = []

    # Stop earlier if no changes
    if early_stopping:
        callbacks.append(
            lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False)
        )

    # Log every X-th line
    if log_evaluation:
        callbacks.append(lgb.log_evaluation(log_evaluation))

    if record_evaluation is not None:
        assert (
                type(record_evaluation) == dict
        ), "´record_evaluation´ has to be dictionary"
        callbacks.append(lgb.record_evaluation(record_evaluation))

    return callbacks


def ztransform2(Y, param):
    return (np.sqrt(Y) - np.sqrt(param)) / np.sqrt(Y + param)


def inverseZ2(Z, param, epsilon=1e-6):
    Z2 = np.minimum(Z, 1 - epsilon)
    Z2 = np.maximum(Z2, -1 + epsilon)
    result = -1 - Z2 * np.sqrt(2 - Z2 ** 2)
    result = result / (Z2 ** 2 - 1)
    result = param * result ** 2
    return result


def _fix_df_train_issues(df):
    rr = df[
        (df["state_abb"] == "NM")
        & (df["county"].str.contains("ana county", case=False))
        ]
    rr = rr["county"].value_counts()
    assert rr.shape[0] == 1, "should only have one county here"
    df["county"] = df["county"].str.replace(rr.index[0], "Dona Ana County")

    return df


def _add_location(df_train):
    df_location = read_df('cfips_location.csv', 'usa-counties-coordinates')

    def rot(df):
        for angle in [15, 30, 45]:
            df[f'rot_{angle}_x'] = (np.cos(np.radians(angle)) * df['lat']) + \
                                   (np.sin(np.radians(angle)) * df['lng'])

            df[f'rot_{angle}_y'] = (np.cos(np.radians(angle)) * df['lat']) - \
                                   (np.sin(np.radians(angle)) * df['lng'])

        return df

    f_location = rot(df_location)

    return pd.merge(df_train, f_location, 'left', 'cfips')


def _fix_df_train(df_train):
    df = df_train.copy()
    df_census = read_df("census_starter.csv")

    # Add year
    df["first_day_of_month"] = pd.to_datetime(df["first_day_of_month"])
    df["year"] = df["first_day_of_month"].dt.year.astype(int)

    # Add df_census to df
    cols = list(df_census.columns)
    cols.remove("cfips")

    t0 = df_census.melt("cfips", cols)
    t0["year"] = t0["variable"].str.split("_").str[-1].astype(int)
    t0["variable_name"] = t0["variable"].str.rsplit("_", expand=False, n=1).str[0]

    t1 = pd.pivot_table(t0, "value", ["cfips", "year"], "variable_name").reset_index()

    # Census data is lagging 2 years
    t1["year"] = t1["year"] + 2

    df = pd.merge(df, t1, "left", left_on=["cfips", "year"], right_on=["cfips", "year"])

    # Add month
    df["month"] = df["first_day_of_month"].dt.month

    # Add location data
    df = _add_location(df)

    return df


def _loop_new_cols(
        df: pd.DataFrame,
        df_mapped_feature: pd.DataFrame,
        f: callable,
        target_col: str,
        groupby_col: str = None,
) -> pd.DataFrame:
    res = []
    for idx, row in df_mapped_feature.iterrows():
        r = f(df, idx, target_col, round(row["params"]), groupby_col)
        res.append(r)

    return pd.concat(res, axis=1)


states_abb = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}
states = {y: x for x, y in states_abb.items()}
