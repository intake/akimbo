import awkward as ak  # noqa
import ray.data as rd

from akimbo.mixin import Accessor

# ray.data.Dataset.zip and .map_batches


class RayAccessor(Accessor):
    dataframe_type = rd.Dataset
    series_type = None  # only has "dataframe like"

    @classmethod
    def _to_output(cls, data: rd.Dataset):
        return data.to_pandas()  # automatically has arrow types
