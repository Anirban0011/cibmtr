from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from data_pps.constants import constants


class CATBoost:
    def __init__(self, FOLDS=10):
        self.FOLDS = FOLDS
        self.kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    def __call__(self, train: pd.DataFrame, test: pd.DataFrame):
        FOLDS = self.FOLDS
        kf = self.kf
        oof_cat = np.zeros(len(train))
        pred_cat = np.zeros(len(test))

        for i, (train_index, test_index) in enumerate(kf.split(train)):

            print("#" * 25)
            print(f"### Fold {i+1}")
            print("#" * 25)

            FEATURES, CATS = constants(train)()

            x_train = train.loc[train_index, FEATURES].copy()
            y_train = train.loc[train_index, "y"]
            x_valid = train.loc[test_index, FEATURES].copy()
            y_valid = train.loc[test_index, "y"]
            x_test = test[FEATURES].copy()

            model_cat = CatBoostRegressor(
                task_type="GPU",
                learning_rate=0.1,
                grow_policy="Lossguide",
                # early_stopping_rounds=25,
            )
            model_cat.fit(
                x_train,
                y_train,
                eval_set=(x_valid, y_valid),
                cat_features=CATS,
                verbose=250,
            )

            # INFER OOF
            oof_cat[test_index] = model_cat.predict(x_valid)
            # INFER TEST
            pred_cat += model_cat.predict(x_test)

        # COMPUTE AVERAGE TEST PREDS
        pred_cat /= FOLDS

        return oof_cat, pred_cat
