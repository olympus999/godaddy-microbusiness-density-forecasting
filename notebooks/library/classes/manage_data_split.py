import pandas as pd
import numpy as np
from . import ManageFeatures
import lightgbm as lgb


class ManageDataSplit:
    def __init__(self, df: pd.DataFrame):
        self.constant = None
        self.df = df.copy().set_index("row_id")

        self._train_idx = None
        self._val_idx = None
        self._test_idx = None

        self._set_split_idx()

    def split_data(self, df_features: pd.DataFrame, df_target: pd.DataFrame, objective: str) -> dict:
        idx_wo_nan = df_features.dropna().index.intersection(df_target.dropna().index)

        # Filter with filterin nan rows
        train_filter = (df_features.index.isin(self._train_idx)) & (
            df_features.index.isin(idx_wo_nan)
        )
        val_filter = (df_features.index.isin(self._val_idx)) & (
            df_features.index.isin(idx_wo_nan)
        )
        test_filter = (df_features.index.isin(self._test_idx)) & (
            df_features.index.isin(idx_wo_nan)
        )

        self.constant = 1
        if objective == 'mape':
            self.constant = 1

        return {
            "train": {
                "data": df_features[train_filter],
                "label": df_target[train_filter],
            },
            "val": {
                "data": df_features[val_filter],
                "label": df_target[val_filter],
            },
            "test": {
                "data": df_features[test_filter],
                "label": df_target[test_filter],
            },
        }

    def get_model_input(
            self, manage_features: ManageFeatures, df_train, objective=None, bayes_kwargs: dict = None
    ):
        df_features = manage_features.generate_features(bayes_kwargs)
        df_target = manage_features.generate_target(df_train)

        dict_split_df = self.split_data(df_features, df_target, objective)

        lgb_train = lgb.Dataset(**dict_split_df["train"], free_raw_data=False)
        lgb_eval = lgb.Dataset(
            **dict_split_df["val"], reference=lgb_train, free_raw_data=False
        )
        lgb_test = lgb.Dataset(
            **dict_split_df["test"], reference=lgb_train, free_raw_data=False
        )

        model_params = manage_features.get_model_params(bayes_kwargs)

        return lgb_train, lgb_eval, lgb_test, model_params

    def get_train_data(self):
        return self.df.loc[self._train_idx]

    def _set_split_idx(
            self,
            train_size: float = 0.70,
            val_size: float = 0.20,
            test_size: float = 0.10,
            date_col: str = "first_day_of_month",
    ):
        dates = np.sort(self.df[date_col].unique())

        dates_idx_train_end = int(dates.shape[0] * train_size)
        dates_idx_val_end = int(dates.shape[0] * (train_size + val_size))

        dates_train = dates[0:dates_idx_train_end]
        dates_val = dates[dates_idx_train_end:dates_idx_val_end]
        dates_test = dates[dates_idx_val_end:]

        self._train_idx = self.df[self.df[date_col].isin(dates_train)].index
        self._val_idx = self.df[self.df[date_col].isin(dates_val)].index
        self._test_idx = self.df[self.df[date_col].isin(dates_test)].index
