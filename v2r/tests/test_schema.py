"""Schema IO round-trip tests."""

import numpy as np
import pandas as pd
import pytest

from v2r.schema.io import poses_arrays, poses_df, write_table, read_table, SchemaError
from v2r.schema.models import SourceTag


def test_poses_parquet_roundtrip(tmp_path):
    n = 5
    t = np.arange(n) / 30.0
    frames = np.arange(n)
    T = np.tile(np.eye(4), (n, 1, 1))
    conf = np.full(n, 0.9)
    valid = np.ones(n, dtype=bool)
    df = poses_df(t, frames, T, conf, valid, SourceTag.synthesized.value)
    path = tmp_path / "poses.parquet"
    write_table(df, path)
    back = read_table(path)
    arr = poses_arrays(back)
    assert len(arr["t"]) == n
    assert arr["source"][0] == SourceTag.synthesized.value


def test_invalid_conf_rejected(tmp_path):
    df = pd.DataFrame({"t": [0.0], "conf": [1.5], "valid": [True], "source": ["synthesized"]})
    with pytest.raises(SchemaError):
        write_table(df, tmp_path / "bad.parquet")
