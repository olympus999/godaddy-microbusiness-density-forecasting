import pandas as pd
from traitlets import HasTraits, Unicode, Callable, Tuple, Float, List, Dict, default, validate, TraitError
from .trait import DataFrame

feature_objects = {}


class Feature(HasTraits):
    # During init

    f_col = Unicode()
    f = Callable()

    bound = Tuple(Float(), Float())

    enabled_bounds = List(bound, minlen=1)
    params_bounds = List(bound)

    # After init

    _enabled_dict = Dict(key_trait=Unicode(), value_trait=bound)
    _params_dict = Dict(key_trait=Unicode(), value_trait=bound)

    relations = Dict(key_trait=Unicode(), value_trait=Unicode())

    _df_enabled_params = DataFrame()

    def __init__(self, f_col, f, df, enabled_bounds=None, params_bounds=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_col = f_col
        self.f = f
        self.df = df
        if enabled_bounds:
            self.enabled_bounds = enabled_bounds
        if params_bounds:
            self.params_bounds = params_bounds

        self._add_self_to_global_feature_objects()
        self._make_enable_and_params_dict()
        self._make_df_enable_params()

        self._kwargs = kwargs

    def get_relations(self):
        return self._relations

    def get_enabled_dict(self):
        return self._enabled_dict

    def get_params_dict(self):
        return self._params_dict

    def get_df_enable_params(self):
        return self._df_enabled_params

    def _add_self_to_global_feature_objects(self):
        """
        This global list is used to pick up features by ´ManageFeatures´
        """
        global feature_objects
        feature_objects[self.f_col] = self

    def _set_relations(self, d):
        self._relations = d

    def _make_df_enable_params(self):
        d = self.get_relations()
        relation_original = d
        relation_flipped = {v: k for k, v in d.items()}

        # Get relations
        df_relation = pd.DataFrame.from_dict(
            {**relation_original, **relation_flipped},
            orient="index",
            columns=["relation"],
        )

        # Bounds
        df_bounds = pd.DataFrame.from_dict(
            {
                **self._enabled_dict,
                **self._params_dict,
            },
            orient="index",
            columns=["min", "max"],
        )

        # Kind column separating enabled/params
        df_bounds.loc[self._enabled_dict.keys(), "kind"] = "enabled"
        df_bounds.loc[self._params_dict.keys(), "kind"] = "params"

        # Join
        df = pd.merge(df_bounds, df_relation, "left", left_index=True, right_index=True)

        # Add f_col
        df["f_col"] = self.f_col

        # Fill relations where there are none 
        # when feature has no params, but bayes wants to switch it on/off

        df['relation'] = df['relation'].fillna(df.index.to_series())

        self._df_enabled_params = df

    def _make_enable_and_params_dict(self):
        """
        Create 2 variables:
        1) self._enabled_dict
        2) self._params_dict
        """
        self._enabled_keys = []
        self._params_keys = []
        for kind in ["enabled", "params"]:
            bounds = getattr(self, "{}_bounds".format(kind))

            d = {}
            key_f_col = "{kind}_{f_col}_{idx}"
            for idx, bound in enumerate(bounds):
                if len(bounds) < 2:
                    idx = ""
                key = key_f_col.format(kind=kind, f_col=self.f_col, idx=idx)
                d[key] = bound

                getattr(self, "_{}_keys".format(kind)).append(key)

            setattr(self, "_{}_dict".format(kind), d)
        # Set relations for all
        d = dict(zip(self._enabled_keys, self._params_keys))
        self._set_relations(d)

    @default("enabled_bounds")
    def _default_value(self):
        return [(0, 1)]

    @validate("params_bounds")
    def _valid_params_bounds(self, proposal):
        params_bounds = proposal["value"]

        if params_bounds is not None:
            len_params_bounds = len(params_bounds)
            len_enabled_bounds = len(self.enabled_bounds)
            if len_params_bounds != len_enabled_bounds:
                raise TraitError(
                    "If defined, ´params_bounds´ ({}) should match in length with ´enabled_bounds´ ({})".format(
                        len_params_bounds, len_enabled_bounds
                    )
                )

        return proposal["value"]

    @validate("enabled_dict")
    def _valid_enabled_dict(self, proposal):
        for key, value in proposal["value"].items():
            bound_min = value[0]
            bound_max = value[1]

            if (bound_min < 0 or bound_min > 1) or (bound_max < 0 or bound_max > 1):
                raise TraitError(
                    "For key ´{}´ bound values have to be between 0 and 1 (both included). Values given: ´{}´".format(
                        key, value
                    )
                )

            if bound_min > bound_max:
                raise TraitError(
                    "Bound min cannot be bigger than max. Values given: ´{}´".format(
                        value
                    )
                )

        return proposal["value"]
