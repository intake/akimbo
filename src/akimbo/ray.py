import awkward as ak  # noqa
import ray.data

from akimbo.mixin import Accessor

# ray.data.Dataset.zip and .map_batches


class RayAccessor(Accessor):
    dataframe_type = ray.data.Dataset
    series_type = None  # only has "dataframe like"

    @classmethod
    def _to_output(cls, data: ray.data.Dataset):
        return data.to_pandas()
