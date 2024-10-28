from typing import Iterable

import pyspark
from pyspark.sql.functions import pandas_udf

from akimbo.mixin import Accessor
from akimbo.pandas import pd

# https://docs.databricks.com/en/udf/pandas.html


class SparkAccessor(Accessor):
    def __dir__(self) -> Iterable[str]:
        return dir(pd.DataFrame.ak)

    def __getattr__(self, item):
        @pandas_udf(returnType=pyspark.sql.types.BooleanType())
        def run(x: pd.Series) -> pd.Series:
            from akimbo.pandas import pd  # noqa

            out = x.ak.is_none()
            return out

        return self._obj.select(run("x"))


@property  # type:ignore
def ak_property(self):
    return SparkAccessor(self)


pyspark.sql.DataFrame.ak = ak_property
