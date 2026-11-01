import pandas as pd
import pytest

from src import eda
from src.preprocess import add_derived, clean, frame_from_lines

LINES = [
    "02;250021111111111;8991111111;0261111111;20261001100000;0;",
    "02;250021111111111;8992222222;0262222222;20261001100030;12;",
    "01;;8421111111;8991111111;20261001110000;40;",
    "01;;8991111111;0261111111;20261001100000;0;",
    "03;;8421111111;2101111111;20261002090000;15;0263333333",
    "02;250021111111111;7771111111;0261111111;20261002090500;3;",
]


@pytest.fixture(scope="module")
def calls():
    c, _, _ = clean(frame_from_lines(LINES))
    return add_derived(c)


def test_overview(calls):
    o = eda.overview(calls)
    assert o["записей"] == 6
    assert o["уникальных IMSI"] == 1


def test_calls_by_day_hour(calls):
    p = eda.calls_by_day_hour(calls)
    assert p.to_numpy().sum() == 6 and p.loc[pd.Timestamp("2026-10-01"), 10] == 3


def test_type_semantics(calls):
    t = eda.type_semantics(calls)
    assert t.loc["03", "share_with_c"] == 1
    assert t.loc["02", "share_with_imsi"] == 1


def test_mirrored_records(calls):
    m = eda.mirrored_records(calls)
    assert m["из них с разными типами записей"] == 1


def test_unknown_codes(calls):
    assert eda.unknown_codes(calls).to_dict() == {"777": 1}


def test_sim_check(calls):
    s = eda.sim_check(calls)
    assert s.loc[3, "номеров на IMSI"] == 1


def test_top_callers(calls):
    top = eda.top_callers(calls, n=2)
    assert set(top.index) == {"8991111111", "8421111111"}
    assert top["n_out"].tolist() == [2, 2]
    assert str(top.loc["8991111111", "operator"]) == "Барбетта"


def test_run_writes_tables_and_figures(calls, tmp_path):
    path = tmp_path / "calls.parquet"
    calls.to_parquet(path, index=False)
    eda.run(path, tmp_path / "eda")
    for name in ["overview.csv", "top_callers.csv", "calls_day_hour.png", "duration_hist.png"]:
        assert (tmp_path / "eda" / name).exists(), name
