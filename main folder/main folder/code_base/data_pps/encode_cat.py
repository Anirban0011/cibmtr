import pandas as pd
from .constants import constants

# from utils.methods import ReducedPrecision, Categorize


class EncodeCat:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def __call__(self):
        train = self.train
        test = self.test
        FEATURES, CATS = constants()()
        for c in FEATURES:
            if train[c].dtype == "object":
                train[c] = train[c].fillna("NAN")
                test[c] = test[c].fillna("NAN")
        combined = pd.concat([train, test], axis=0, ignore_index=True)

        # print("Combined data shape:", combined.shape )
        # LABEL ENCODE CATEGORICAL FEATURES
        for c in FEATURES:
            # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
            if c in CATS:
                combined[c], _ = combined[c].factorize()
                combined[c] -= combined[c].min()
                combined[c] = combined[c].astype("int32")
                # combined[c] = Categorize()(c, combined)
                combined[c] = combined[c].astype("category")

            # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
            else:
                if combined[c].dtype == "float64":
                    combined[c] = combined[c].astype("float32")
                if combined[c].dtype == "int64":
                    combined[c] = combined[c].astype("int32")
                # combined[c] = ReducedPrecision()(c, combined)

        train = combined.iloc[: len(train)].copy()
        test = combined.iloc[len(train) :].reset_index(drop=True).copy()

        return train, test
