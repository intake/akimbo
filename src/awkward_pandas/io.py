import awkward._v2 as ak
import pandas as pd

import awkward_pandas


def read_parquet(url, extract=True, **kwargs):
    ds = ak.from_parquet(url, **kwargs)
    s = pd.Series(awkward_pandas.AwkwardExtensionArray(ds))
    if extract is False:
        return s
    return s.ak.to_columns(cull=True)
