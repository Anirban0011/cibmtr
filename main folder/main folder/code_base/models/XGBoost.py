from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
from code_base.data_pps.constants import constants


class XGBoost:
    def __init__(self, FOLDS=10):
        self.FOLDS = FOLDS
        self.kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    def __call__(self, train: pd.DataFrame, test: pd.DataFrame):
        kf = self.kf
        FOLDS = self.FOLDS
        oof_xgb = np.zeros(len(train))
        pred_xgb = np.zeros(len(test))

        for i, (train_index, test_index) in enumerate(kf.split(train)):

            print("#" * 25)
            print(f"### Fold {i+1}")
            print("#" * 25)

            FEATURES, _ = constants()()

            x_train = train.loc[train_index, FEATURES].copy()
            y_train = train.loc[train_index, "y"]
            x_valid = train.loc[test_index, FEATURES].copy()
            y_valid = train.loc[test_index, "y"]
            x_test = test[FEATURES].copy()

            model_xgb = XGBRegressor(
                device="cuda",
                max_depth=3,
                colsample_bytree=0.5,
                subsample=0.8,
                n_estimators=2000,
                learning_rate=0.02,
                enable_categorical=True,
                min_child_weight=80,
                # early_stopping_rounds=25,
            )
            model_xgb.fit(
                x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=500
            )

            # INFER OOF
            oof_xgb[test_index] = model_xgb.predict(x_valid)
            # INFER TEST
            pred_xgb += model_xgb.predict(x_test)

        # COMPUTE AVERAGE TEST PREDS
        pred_xgb /= FOLDS

        return oof_xgb, pred_xgb