from lightgbm import LGBMRegressor
import lightgbm as lgb
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
        oof_lgb = np.zeros(len(train))
        pred_lgb = np.zeros(len(test))

        for i, (train_index, test_index) in enumerate(kf.split(train)):

            print("#" * 25)
            print(f"### Fold {i+1}")
            print("#" * 25)

            FEATURES, _ = constants(train)()

            x_train = train.loc[train_index, FEATURES].copy()
            y_train = train.loc[train_index, "y"]
            x_valid = train.loc[test_index, FEATURES].copy()
            y_valid = train.loc[test_index, "y"]
            x_test = test[FEATURES].copy()

            model_lgb = LGBMRegressor(
                device="gpu",
                max_depth=3,
                colsample_bytree=0.4,
                # subsample=0.9,
                n_estimators=2500,
                learning_rate=0.02,
                objective="regression",
                verbose=-1,
                # early_stopping_rounds=25,
            )
            model_lgb.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
            )

            # INFER OOF
            oof_lgb[test_index] = model_lgb.predict(x_valid)
            # INFER TEST
            pred_lgb += model_lgb.predict(x_test)

        # COMPUTE AVERAGE TEST PREDS
        pred_lgb /= FOLDS

        return oof_lgb, pred_lgb
