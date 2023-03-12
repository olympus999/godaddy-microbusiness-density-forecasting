from .utils import _loop_new_cols
import pandas as pd
import numpy as np


def f_rolling_mean(
        df: pd.DataFrame,
        df_mapped_feature: pd.DataFrame,
        target_col: str,
        f_col: str,
        groupby_col: str,
        **kwargs
):
    def _f(df: pd.DataFrame, col: str, target_col: str, window: int, groupby_col: str):
        assert window > 0, "Window has to be above 0"
        rolling_mean = (
            df.set_index('row_id')
            .sort_values([groupby_col, "first_day_of_month"])
            .groupby([groupby_col])[target_col]
            .rolling(window, closed="left")
            .mean()
            .rename(col)
            .reset_index(drop=True)
        )
        rolling_mean.index = df["row_id"]

        return rolling_mean

    return _loop_new_cols(df, df_mapped_feature, _f, target_col, groupby_col)


def f_shifted(
        df: pd.DataFrame,
        df_mapped_feature: pd.DataFrame,
        target_col: str,
        f_col: str,
        **kwargs
) -> pd.DataFrame:
    def _f(df: pd.DataFrame, col: str, target_col: str, shift: int, *args):
        assert shift >= 1, "lower shift leads to leakage of target variable"
        shifted = (
            df.sort_values(["cfips", "first_day_of_month"])
            .groupby(["cfips"])[target_col]
            .shift(shift)
            .rename(col)
            .reset_index(drop=True)
        )
        shifted.index = df["row_id"]

        return shifted

    return _loop_new_cols(df, df_mapped_feature, _f, target_col, **kwargs)


def add_categorical_feature(df: pd.DataFrame, f_col: str, **kwargs):
    return df[['row_id', f_col]].set_index('row_id').astype('category')


def add_numerical_feature(df: pd.DataFrame, f_col: str, **kwargs):
    df[f_col] = df[f_col].astype(float)
    return df[['row_id', f_col]].set_index('row_id')


def time_arrow(df: pd.DataFrame, f_col: str, **kwargs):
    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    seconds_since = df["first_day_of_month"].astype("int64") // 1e9
    df[f_col] = normalize_data(seconds_since)

    return df[['row_id', f_col]].set_index('row_id')


def add_feature_targets_groupby_stats(
    df,
    f_col,
    # new_col_template="{}_target_{}",
    agg_function=None,
    # agg_functions=["mean", "std", "median"],
    train_idx=None,
    groupby_col=None,
    col=None,
    **kwargs
):
    # df = df.copy()
    t0 = df.groupby(groupby_col)[col].agg(agg_function).rename(f_col).reset_index()
    t0 = pd.merge(df, t0, 'left', groupby_col).set_index('row_id')[f_col]
    return t0


def f_microbusiness_pct_change(
        df: pd.DataFrame,
        f_col: str,
        **kwargs
):
    df = df.sort_values(["cfips", "first_day_of_month"])
    df["microbusiness_density_shift_-1"] = df.groupby(["cfips"])[
        "microbusiness_density"
    ].shift(-1)
    df[f_col] = df.groupby("cfips")[
        "microbusiness_density_shift_-1"
    ].pct_change()
    df.replace([np.inf, -np.inf], 1000, inplace=True)

    idx = df[
        (df[f_col].isna())
        & (df["microbusiness_density_shift_-1"] == 0)
        ].index

    df.loc[idx, f_col] = 0

    return df.set_index('row_id')[[f_col]]


def f_microbusiness_density_diff(
        df: pd.DataFrame,
        f_col: str,
        **kwargs
):
    df[f_col] = df["microbusiness_density"].shift(1) - df[
        "microbusiness_density"
    ].shift(2)

    return df.set_index('row_id')[[f_col]]