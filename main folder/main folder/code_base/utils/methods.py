import pandas as pd


class ReducedPrecision:
    def __call__(self, c, combined: pd.DataFrame):
        if combined[c].dtype == "float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype == "int64":
            combined[c] = combined[c].astype("int32")
        return combined[c]


class Categorize:
    def __call__(self, c, combined: pd.DataFrame):
        combined[c], _ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        return combined[c]
