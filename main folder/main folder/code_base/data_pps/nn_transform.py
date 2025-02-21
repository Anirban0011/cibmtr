import pandas as pd
import numpy as np

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
                train[c] = train[c].fillna("NAN")
                test[c] = test[c].fillna("NAN")
                CATS.append(c)
            elif not "age" in c:
                train[c] = train[c].astype("str")
                test[c] = test[c].astype("str")
                CATS.append(c)
        CAT_SIZE = []
        CAT_EMB = []
        NUMS = []

        combined = pd.concat([train,test],axis=0,ignore_index=True)

        for c in FEATURES:
            if c in CATS:
                # LABEL ENCODE
                combined[c],_ = combined[c].factorize()
                combined[c] -= combined[c].min()
                combined[c] = combined[c].astype("int32")
                #combined[c] = combined[c].astype("category")

                n = combined[c].nunique()
                mn = combined[c].min()
                mx = combined[c].max()
                CAT_SIZE.append(mx+1)
                CAT_EMB.append( int(np.ceil( np.sqrt(mx+1))) )
            else:
                if combined[c].dtype=="float64":
                    combined[c] = combined[c].astype("float32")
                if combined[c].dtype=="int64":
                    combined[c] = combined[c].astype("int32")

                m = combined[c].mean()
                s = combined[c].std()
                combined[c] = (combined[c]-m)/s
                combined[c] = combined[c].fillna(0)

                NUMS.append(c)

        train = combined.iloc[:len(train)].copy()
        test = combined.iloc[len(train):].reset_index(drop=True).copy()

        return train, test, CAT_SIZE, CAT_EMB, NUMS