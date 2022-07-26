def test_add(int_series, int_raw_array):
    s = int_series
    s2 = s + 1
    assert s2.dtype == "awkward"
    assert s2.tolist() == (int_raw_array + 1).tolist()
