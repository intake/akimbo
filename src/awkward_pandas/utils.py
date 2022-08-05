import awkward._v2 as ak
import pandas as pd

from awkward_pandas.array import AwkwardExtensionArray


def merge_columns(df, columns, name=None):
    if name is None:
        name = "".join(columns)
    a = ak.zip({c: df[c] for c in columns})
    s = pd.Series(AwkwardExtensionArray(a))
    out = {c: df[c] for c in df.columns if c not in columns}
    out[name] = s
    return pd.DataFrame(out)
