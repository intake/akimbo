import awkward._v2 as ak
import awkward_pandas

import pandas as pd

def read_parquet(url, **kwargs):
    ds = ak.from_parquet(url, **kargs)
    return pd.Series(awkward_pandas.AwkwardArray(ds))
