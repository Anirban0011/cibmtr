from lifelines import KaplanMeierFitter, BreslowFlemingHarringtonFitter
import pandas as pd
import numpy as np


class TargetTransform:
    def __init__(self):
        pass

    """
    train["y"] = return
    """

    def basic_transform(self, train: pd.DataFrame):
        train["y"] = train.efs_time.values
        mx = train.loc[train.efs == 1, "efs_time"].max()
        mn = train.loc[train.efs == 0, "efs_time"].min()
        train.loc[train.efs == 0, "y"] = (
            train.loc[train.efs == 0, "y"] + mx - mn
        )
        train.y = train.y.rank()
        train.loc[train.efs == 0, "y"] += len(train) // 2
        train.y = train.y / train.y.max()
        return train.y

    def basic_transform2(self, train: pd.DataFrame):
        train["y"] = train.efs_time.values
        mx = train.loc[train.efs == 1, "efs_time"].max()
        mn = train.loc[train.efs == 0, "efs_time"].min()
        train.loc[train.efs == 0, "y"] = (
            train.loc[train.efs == 0, "y"] + mx - mn
        )
        train.y = train.y.rank()
        train.loc[train.efs == 0, "y"] += 2 * len(train)
        train.y = train.y / train.y.max()
        train.y = np.log(train.y)
        train.y -= train.y.mean()
        train.y *= -1.0
        return train.y

    def KMF_transform(
        self, train: pd.DataFrame, time_col="efs_time", event_col="efs"
    ):
        kmf = KaplanMeierFitter()
        kmf.fit(train[time_col], train[event_col])
        y = kmf.survival_function_at_times(train[time_col]).values
        return y

    def BFHF_transform(
        self, train: pd.DataFrame, time_col="efs_time", event_col="efs"
    ):
        bfhf = BreslowFlemingHarringtonFitter()
        bfhf.fit(train[time_col], train[event_col])
        y = bfhf.survival_function_at_times(train[time_col]).values
        return y
