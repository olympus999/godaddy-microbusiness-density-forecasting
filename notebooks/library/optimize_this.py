import warnings

import lightgbm as lgb
import numpy as np

from .utils import smape


def optimize_this(
        objective,
        pbounds,
        manage_data_split,
        manage_features,
        df_train,
        build_callbacks, print_loss=True, return_booster=False, train=True, target_shift=0, **bayes_kwargs):
    try:
        # Make sure bayes_kwargs are inside the bounds
        for key, _tuple in pbounds.items():
            params = bayes_kwargs
            if params[key] < _tuple[0]:
                # print(params[key], _tuple)
                params[key] = _tuple[0]
            if params[key] > _tuple[1]:
                # print(params[key], _tuple)
                params[key] = _tuple[1]

        input_bayes_kwargs = bayes_kwargs.copy()

        # print(params)
        # print(bayes_kwargs)

        lgb_train, lgb_eval, lgb_test, model_params, df_features, df_target = manage_data_split.get_model_input(
            manage_features, df_train, objective=objective, target_shift=target_shift, bayes_kwargs=bayes_kwargs
        )
        # print('train{}'.format(lgb_train.data.shape), 'test{}'.format(lgb_test.data.shape), 'eval{}'.format(lgb_eval.data.shape))

        # callbacks = build_callbacks()
        callbacks = build_callbacks(early_stopping=300)

        model_params = {
            **model_params,
            **{
                "boosting_type": "gbdt",
                "objective": objective,
                # "metric": "None",
                "metric": "mae",
                "first_metric_only": True,
                "num_threads": 6,
                "verbose": -1,
            },
        }

        # Can be used to supress warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            gbm = None
            if train:
                gbm = lgb.train(
                    model_params,
                    lgb_train,
                    callbacks=callbacks,
                    valid_sets=[lgb_eval],
                    # feval=smape,
                )

                pred = gbm.predict(lgb_test.data)
                # # loss = smape(pred, lgb_test.label)
                loss = smape(np.expm1(pred), np.expm1(lgb_test.label))

            if print_loss:
                print(objective, 'np.exp', 'train{}'.format(lgb_train.data.shape), 'test{}'.format(lgb_test.data.shape),
                      round(-loss[1], 4), 'eval{}'.format(lgb_eval.data.shape),
                      round(-gbm.best_score["valid_0"]["l1"]*100, 4))
                # print(objective, 'np.exp', 'train{}'.format(lgb_train.data.shape), 'test{}'.format(lgb_test.data.shape),
                #       round(-loss[1], 4), 'eval{}'.format(lgb_eval.data.shape),
                #       round(-gbm.best_score["valid_0"]["l1"], 4))
                # print(objective, 'np.exp', 'train{}'.format(lgb_train.data.shape), 'test{}'.format(lgb_test.data.shape),
                #       round(-loss[1], 4), 'eval{}'.format(lgb_eval.data.shape),
                #       round(-gbm.best_score["valid_0"]["smape"], 4))
                # print(objective, 'np.exp', 'train{}'.format(lgb_train.data.shape), 'test{}'.format(lgb_test.data.shape),
                #       round(-loss[1], 4), 'eval{}'.format(lgb_eval.data.shape),
                #       gbm.best_score['valid_0']['l1'])
                # print('train', lgb_train.data.shape, 'eval', lgb_eval.data.shape, 'test', lgb_test.data.shape)
            if return_booster:
                return gbm, lgb_train, lgb_eval, lgb_test, model_params, callbacks, df_features, df_target
            else:
                return -gbm.best_score['valid_0']['l1']
                # return gbm.best_score['valid_0']['smape']
                # return -loss[1]
    except Exception as e:
        print(input_bayes_kwargs)
        raise e
