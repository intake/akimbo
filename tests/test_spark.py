import pytest

pd = pytest.importorskip("pandas")
pyspark = pytest.importorskip("pyspark")
import akimbo.spark


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.appName("test").getOrCreate()


def test1(spark):
    x = pd.Series([1, 2, 3])
    df = spark.createDataFrame(pd.DataFrame(x, columns=["x"]))
    out = df.ak.is_none.collect()
    assert out.tolist() == [False, False, False]
