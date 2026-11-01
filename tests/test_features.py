import numpy as np
import pandas as pd
import pytest

from src.features import build_features, window_counts
from src.preprocess import add_derived, clean, frame_from_lines

X = "8991111111"
X2 = "8992222222"
Y = "0261111111"
Z = "2101111111"
W = "8421111111"
IMSI = "250021111111111"

LINES = [
    f"02;{IMSI};{X};{Y};20261001100000;0;",
    f"02;{IMSI};{X};0262222222;20261001100030;1;",
    f"02;{IMSI};{X};0263333333;20261001100100;5;",
    f"02;{IMSI};{X};0264444444;20261001100300;30;",
    f"02;{IMSI};{X2};{Z};20261001120000;200;",
    f"02;250012222222222;{Y};{X};20261001110000;40;",
    f"02;250012222222222;{Y};{Z};20261002230000;60;",
    f"02;250012222222222;{Y};{Z};20261003020000;abc;",
    f"03;;{W};{Z};20261001130000;15;{X}",
]


@pytest.fixture(scope="module")
def feats():
    calls, _, _ = clean(frame_from_lines(LINES))
    return build_features(add_derived(calls))


def test_window_counts_basic():
    gid = np.array([0, 0, 0, 1, 1])
    t = np.array([0, 100, 400, 0, 10])
    assert window_counts(gid, t, 300).tolist() == [2, 1, 1, 2, 1]


def test_every_number_gets_a_row(feats):
    assert {X, X2, Y, Z, W, "0262222222"} <= set(feats.index)
    assert (feats.index.str.len() == 10).all()


def test_outgoing_counts_and_peaks(feats):
    x = feats.loc[X]
    assert x["n_out"] == 4 and x["uniq_b"] == 4 and x["repeat_share"] == 0
    assert x["peak_5m"] == 4
    assert x["peak_1h"] == 4
    assert x["active_days"] == 1 and x["max_calls_day"] == 4


def test_duration_shares(feats):
    x = feats.loc[X]
    assert x["share_dur0"] == pytest.approx(0.25)
    assert x["share_le2"] == pytest.approx(0.5)
    assert x["share_short"] == pytest.approx(0.75)
    assert x["dur_median"] == pytest.approx(3)


def test_unknown_duration_is_not_counted_as_zero(feats):
    y = feats.loc[Y]
    assert y["share_dur_missing"] == pytest.approx(1 / 3)
    assert y["share_short"] == 0
    assert y["dur_mean"] == pytest.approx(50)


def test_time_features(feats):
    y = feats.loc[Y]
    assert y["active_days"] == 3
    assert y["share_night"] == pytest.approx(1 / 3)
    assert y["share_weekend"] == pytest.approx(1 / 3)
    assert feats.loc[X, "share_weekend"] == 0
    assert feats.loc[X, "gap_median"] == pytest.approx(30)


def test_callback_share(feats):
    assert feats.loc[X, "callback_share"] == pytest.approx(0.25)
    assert feats.loc[Y, "callback_share"] == pytest.approx(0.5)


def test_sim_features(feats):
    assert feats.loc[X, "n_imsi"] == 1
    assert feats.loc[X, "numbers_per_imsi_max"] == 2
    assert feats.loc[Y, "numbers_per_imsi_max"] == 1


def test_incoming_and_forwarding(feats):
    z = feats.loc[Z]
    assert z["n_out"] == 0 and z["n_in"] == 4 and z["uniq_a"] == 3
    assert z["n_forwarded"] == 1
    assert feats.loc[X, "n_as_forward_target"] == 1
    assert pd.isna(z["share_short"])


def test_operator_and_city(feats):
    assert feats.loc[Y, "operator"] == "Альфа"
    assert feats.loc[W, "is_city"]
    assert not feats["unknown_code"].any()
