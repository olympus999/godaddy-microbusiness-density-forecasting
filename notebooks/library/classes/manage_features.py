from typing import Iterator

import pandas as pd
import numpy as np
from traitlets import HasTraits, Unicode, Tuple, Float, List, Dict

from .trait import DataFrame
from . import Feature


class ManageFeatures(HasTraits):
    _feature_objects = Dict(key_trait=Unicode())
    _df_enabled_params = DataFrame()
    _enabled_keys = List(Unicode())
    _model_pbounds = Dict(key_trait=Unicode(), value_trait=Tuple(Float(), Float()))

    def __init__(self, feature_objects, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_df = None
        self._feature_objects = feature_objects
        # self._target_col = target_col

        self._set_enabled_keys()
        self._set_features_df_enabled_params()

    def set_model_pbounds(self, model_pbounds: dict):
        self._model_pbounds = model_pbounds

    def get_model_params(self, bay_params):
        model_params = {k: v for k, v in bay_params.items() if k in self._model_pbounds}

        model_params["num_iterations"] = round(model_params["num_iterations"])
        model_params["num_leaves"] = round(model_params["num_leaves"])
        model_params['bagging_freq'] = int(np.round(model_params['bagging_freq']))
        model_params['max_depth'] = int(np.round(model_params['max_depth']))
        model_params['min_data_in_leaf'] = int(np.round(model_params['min_data_in_leaf']))

        for key, _tuple in self._model_pbounds.items():
            if model_params[key] < _tuple[0]:
                model_params[key] = _tuple[0]
            if model_params[key] > _tuple[1]:
                model_params[key] = _tuple[1]

        return model_params

    def get_pbounds(self):
        df = self._get_features_df_enabled_params()
        df["pbounds"] = list(zip(df["min"], df["max"]))

        return {**self._model_pbounds, **df["pbounds"].to_dict()}

    def generate_features(self, bay_params: dict):
        # Remove model_pbounds
        df_mapped = self._make_mapped(bay_params)

        res = []
        for feature in self._iter_feature_objects():
            df_mapped_feature = df_mapped[df_mapped["f_col"] == feature.f_col]

            # Some features are used, some not
            if df_mapped_feature.shape[0] != 0:
                r = feature.f(
                    df=feature.df.copy(),
                    df_mapped_feature=df_mapped_feature,
                    f_col=feature.f_col,
                    **feature._kwargs
                )

                assert (
                        r.index.name == "row_id"
                ), "All features need to have ´row_id´ as index"

                res.append(r)
        df_features = pd.concat(res, axis=1)

        return df_features

    def generate_target(self, df_train: pd.DataFrame, shift=0):
        t0 = (
            df_train.sort_values(["cfips", "first_day_of_month"])
            .groupby("cfips")["microbusiness_density"]
            .shift(shift)
        )

        t0.index = df_train["row_id"]

        return t0

    def get_params(self, bay_params: dict):
        # Mapping for columns
        df_mapped = self.get_mapping(bay_params)

        # Params
        idx_params = df_mapped[
            (df_mapped["kind"] == "enabled") & (df_mapped["params"] > 0.5)
            ]["relation"]
        df_params = df_mapped[df_mapped.index.isin(idx_params)]

        return df_params

    def get_mapping(self, bay_params: dict):
        df_enabled_params = self._get_features_df_enabled_params()
        df_bay_params = pd.DataFrame.from_dict(
            bay_params, orient="index", columns=["params"]
        )

        df = pd.merge(
            df_enabled_params, df_bay_params, "outer", left_index=True, right_index=True
        )

        self._df_enabled_params = df_enabled_params.copy()
        self._debug_df = df.copy()

        assert (
                df.shape == df.dropna(subset=["pbounds", "params"]).shape
        ), "There should be no NaN values"

        return df

    def _make_mapped(self, bay_params: dict) -> pd.DataFrame:
        bay_params = {
            k: v for k, v in bay_params.items() if k not in self._model_pbounds
        }

        df_mapped = self.get_params(bay_params)

        return df_mapped

    def _set_enabled_keys(self):
        l = []
        for key, value in self._feature_objects.items():
            l.extend(value._enabled_keys)

        self._enabled_keys = l

    def _get_features_df_enabled_params(self):
        return self._df_enabled_params

    def _set_features_df_enabled_params(self, f="get_df_enable_params"):
        l = []
        # for key, obj in self._feature_objects.items():
        for feature in self._iter_feature_objects():
            l.append(getattr(feature, f)())

        self._df_enabled_params = self._validate_df_duplicate_index_and_concat(l)

    def _validate_df_duplicate_index_and_concat(self, l):
        rows_in_l = 0
        for t in l:
            rows_in_l += t.shape[0]

        df = pd.concat(l)

        assert (
                rows_in_l == df.shape[0]
        ), "Row count should be same. You probably have a duplicate index between diferent ´Feature objects´"

        return df

    def _iter_feature_objects(self) -> Iterator[Feature]:
        for key, obj in self._feature_objects.items():
            yield obj
