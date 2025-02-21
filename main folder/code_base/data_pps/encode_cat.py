import pandas as pd

class EncodeCat():
    def __init__(self, train :pd.DataFrame, test :pd.DataFrame):
        self.train = train
        self.test = test

    def __call__(self):
        train = self.train
        test = self.test
        RMV = ["ID","efs","efs_time","y"]
        FEATURES = [c for c in train.columns if not c in RMV]
        CATS = []
        for c in FEATURES:
            if train[c].dtype=="object":
                CATS.append(c)
                train[c] = train[c].fillna("NAN")
                test[c] = test[c].fillna("NAN")
        combined = pd.concat([train,test],axis=0,ignore_index=True)

        #print("Combined data shape:", combined.shape )
        # LABEL ENCODE CATEGORICAL FEATURES
        for c in FEATURES:

            # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
            if c in CATS:
                combined[c],_ = combined[c].factorize()
                combined[c] -= combined[c].min()
                combined[c] = combined[c].astype("int32")
                combined[c] = combined[c].astype("category")

            # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
            else:
                if combined[c].dtype=="float64":
                    combined[c] = combined[c].astype("float32")
                if combined[c].dtype=="int64":
                    combined[c] = combined[c].astype("int32")

        train = combined.iloc[:len(train)].copy()
        test = combined.iloc[len(train):].reset_index(drop=True).copy()

        return train, test