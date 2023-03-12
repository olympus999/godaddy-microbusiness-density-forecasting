import pandas as pd
import numpy as np
from . import ManageFeatures
import lightgbm as lgb


class ManageDataSplit:
    def __init__(self, df: pd.DataFrame, drop_na=True):
        self.drop_na = drop_na
        self.constant = None
        self.df = df.copy().set_index("row_id")

        self._train_idx = None
        self._val_idx = None
        self._test_idx = None

        self._set_split_idx()

    def split_data(self, df_features: pd.DataFrame, df_target: pd.Series, objective: str, lower_quantile: float,
                   upper_quantile: float) -> dict:
        assert df_target.shape[0] == df_features.shape[
            0], 'df_features.shape[0] {} == df_target.shape[0] {}. Should have same number of columns'.format(
            df_features.shape[0], df_target.shape[0])

        filter_wo_nan = True
        if self.drop_na:
            idx_wo_nan = df_features.dropna().index.intersection(df_target.dropna().index)
            filter_wo_nan = df_features.index.isin(idx_wo_nan)

        filter_lower_quantile = True
        if lower_quantile > 0:
            q_value = df_target.quantile(lower_quantile)
            if ~np.isnan(q_value):
                lower_quantile_idx = df_target[
                    q_value < df_target
                    ].index
                filter_lower_quantile = df_target.index.isin(lower_quantile_idx)

        filter_upper_quantile = True
        if upper_quantile > 0:
            q_value = df_target.quantile(upper_quantile)
            if ~np.isnan(q_value):
                upper_quantile_idx = df_target[
                    q_value > df_target
                    ].index
                filter_upper_quantile = df_target.index.isin(upper_quantile_idx)

        # print('quantiles', upper_quantile, lower_quantile)

        # Filter with filterin nan rows
        train_filter = (df_features.index.isin(self._train_idx)) & (
                filter_wo_nan & filter_lower_quantile & filter_upper_quantile
        )
        val_filter = (df_features.index.isin(self._val_idx)) & (
            filter_wo_nan
        )
        test_filter = (df_features.index.isin(self._test_idx)) & (
            filter_wo_nan
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
            self, manage_features: ManageFeatures, df_train, objective=None, target_shift=0, bayes_kwargs: dict = None
    ):
        upper_quantile = bayes_kwargs.pop("upper_quantile", 1)
        lower_quantile = bayes_kwargs.pop("lower_quantile", 0)

        df_features = manage_features.generate_features(bayes_kwargs)
        df_target = manage_features.generate_target(df_train, shift=target_shift)

        self._df_features = df_features
        self._df_target = df_target
        # print('target_shift', target_shift)

        dict_split_df = self.split_data(df_features, df_target, objective, lower_quantile, upper_quantile)

        lgb_train = lgb.Dataset(**dict_split_df["train"], free_raw_data=False)
        lgb_eval = lgb.Dataset(
            **dict_split_df["val"], reference=lgb_train, free_raw_data=False
        )
        lgb_test = lgb.Dataset(
            **dict_split_df["test"], reference=lgb_train, free_raw_data=False
        )

        model_params = manage_features.get_model_params(bayes_kwargs)

        return lgb_train, lgb_eval, lgb_test, model_params, df_features, df_target

    def get_train_data(self):
        return self.df.loc[self._train_idx]

    def _set_split_idx(
            self,
            train_size: float = 0.70,
            val_size: float = 0.20,
            test_size: float = 0.10,
            date_col: str = "first_day_of_month",
            # upper_quantile: float = None,
            # lower_quantile: float = None,
            **kwargs
    ):
        # print('test')
        dates = np.sort(self.df[date_col].unique())

        dates_idx_train_end = int(dates.shape[0] * train_size)
        dates_idx_val_end = int(dates.shape[0] * (train_size + val_size))

        dates_train = dates[0:dates_idx_train_end]
        dates_val = dates[dates_idx_train_end:dates_idx_val_end]
        dates_test = dates[dates_idx_val_end:]

        # dates_train = ['2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01',
        #                '2019-12-01', '2020-01-01', '2020-02-01', '2020-03-01',
        #                '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01',
        #                '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01',
        #                '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01',
        #                '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
        #                '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01',
        #                '2021-12-01', '2022-01-01', '2022-02-01', '2022-03-01',
        #                '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01',
        #                '2022-08-01', '2022-09-01']
        #
        # dates_val = ['2022-10-01']
        # dates_test = ['2022-11-01', '2022-12-01']

        self._train_idx = self.df[self.df[date_col].isin(dates_train)].index
        self._val_idx = self.df[self.df[date_col].isin(dates_val)].index
        self._test_idx = self.df[self.df[date_col].isin(dates_test)].index
        #
        # lower_quantile_values = 0
        # if lower_quantile is not None:
        #     assert lower_quantile < 0.2, 'lower_quantile cannot be over that'
        #     lower_quantile_values = self.df["microbusiness_density"].quantile(lower_quantile)
        #
        # upper_quantile_value = 0
        # if upper_quantile is not None:
        #     assert upper_quantile > 0.8, 'lower_quantile cannot be under that'
        #     upper_quantile_value = self.df["microbusiness_density"].quantile(upper_quantile)
        #
        # if upper_quantile_value > 0:
        #     self._train_idx = self.df[
        #         upper_quantile_value > self.df["microbusiness_density"]
        #         ].index
        #
        # if lower_quantile_values > 0:
        #     self._train_idx = self.df[
        #         lower_quantile < self.df["microbusiness_density"]
        #         ].index
