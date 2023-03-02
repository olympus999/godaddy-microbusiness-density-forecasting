import pandas as pd
import os

data_path = "../data/"
# boundaries_sub_data_path = "other/boundaries"
# bayesian_run_path = "../data/bayesian_runs/"

def read_df(filename, sub_folder="kaggle", delimiter=",", skiprows=0):
    df = pd.read_csv(os.path.join(data_path, sub_folder, filename), delimiter=delimiter, skiprows=skiprows)

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

    
def _fix_df_train_issues(df):
    rr = df[
        (df["state_abb"] == "NM")
        & (df["county"].str.contains("ana county", case=False))
    ]
    rr = rr["county"].value_counts()
    assert rr.shape[0] == 1, "should only have one county here"
    df["county"] = df["county"].str.replace(rr.index[0], "Dona Ana County")

    return df

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

    return df
    
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