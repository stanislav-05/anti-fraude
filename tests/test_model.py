import pandas as pd
import pytest

from src import model
from src.features import FEATURES

POS = [f"02500000{i:02d}" for i in range(10)]
NEG = [f"21000000{i:02d}" for i in range(60)]
GRAY = "2109999999"
QUIET = "2108888888"


def feature_frame() -> pd.DataFrame:
    rows = {}
    for i, n in enumerate(POS):
        rows[n] = dict(n_out=300 + 50 * i, uniq_b=300 + 50 * i, peak_1h=300 + 40 * i, peak_15m=90 + i,
                       peak_5m=40 + i, dur_median=10 + i % 3, share_short=0.5, share_gap_le60=0.99,
                       calls_per_active_hour=150 + i, active_days=1, callback_share=0.0, share_workhours=0.5)
    for i, n in enumerate(NEG + [GRAY, QUIET]):
        rows[n] = dict(n_out=12 + i, uniq_b=6 + i % 5, peak_1h=2 + i % 3, peak_15m=1 + i % 2, peak_5m=1,
                       dur_median=60 + i, share_short=0.05, share_gap_le60=0.05,
                       calls_per_active_hour=1.5 + i % 4 / 10, active_days=5, callback_share=0.6,
                       share_workhours=0.6)
    rows[QUIET]["n_out"] = 3
    f = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(FEATURES), fill_value=0.0)
    f["operator"] = pd.Categorical(["Альфа"] * len(f))
    f["is_city"] = False
    f["unknown_code"] = False
    return f.rename_axis("number")


def score_frame(f: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=f.index)
    s["risk"] = 0
    s.loc[POS, "risk"] = 100
    s.loc[GRAY, "risk"] = 50
    s["level"] = "низкий"
    s.loc[POS, "level"] = "высокий"
    s.loc[GRAY, "level"] = "проверка"
    s["pattern"] = ""
    s.loc[POS, "pattern"] = "burst"
    s["legit_signs"] = False
    s["n_out"] = f["n_out"]
    return s


@pytest.fixture(scope="module")
def data():
    f = feature_frame()
    return f, score_frame(f)


def test_labels(data):
    f, s = data
    y = model.make_labels(s, f)
    assert (y[POS] == 1).all() and (y[NEG] == 0).all()
    assert pd.isna(y[GRAY]) and pd.isna(y[QUIET])


def test_independent_set_has_no_rule_features(data):
    sets = model.feature_sets(data[0].columns)
    assert not set(model.RULE_FEATURES) & set(sets["independent"])
    assert set(model.RULE_FEATURES) <= set(sets["full"])
    assert not set(model.DROP) & set(sets["full"])


def test_metrics():
    m = model.metrics(pd.Series([1, 1, 0, 0, 0]), pd.Series([0.9, 0.2, 0.7, 0.1, 0.0]).to_numpy())
    assert (m["tp"], m["fn"], m["fp"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == pytest.approx(1 / 3, abs=1e-6)


def test_cross_validation_separates_clear_classes(data):
    f, s = data
    y = model.make_labels(s, f)
    lab = y.notna()
    oof = model.cross_validate(model.prepare_x(f[lab], model.feature_sets(f.columns)["full"]), y[lab])
    assert model.metrics(y[lab], oof)["recall"] == 1.0


def test_shap_factors_format(data):
    f, s = data
    y = model.make_labels(s, f)
    lab = y.notna()
    cols = model.feature_sets(f.columns)["full"]
    m = model.fit(model.prepare_x(f[lab], cols), y[lab])
    text = model.shap_factors(m, model.prepare_x(f.loc[[POS[0]]], cols), top=3)[POS[0]]
    assert text.count(";") == 2 and "=" in text and "(+" in text


def test_isolation_forest_only_active_numbers(data):
    f, _ = data
    iso = model.isolation_forest(f, min_calls=20)
    assert set(iso.index) == set(f.index[f["n_out"] >= 20])
    assert {"if_score", "if_anomaly", "if_factors"} <= set(iso.columns)
    assert iso.loc[POS, "if_score"].min() > iso.drop(index=POS)["if_score"].median()


def test_temporal_check_on_calls():
    """Каждый день - новые номера-залпы, обычные абоненты звонят друзьям, друзья перезванивают."""

    rows = []
    for d in range(1, 6):
        for k in range(2):
            a = f"025{d}{k}00000"
            t0 = pd.Timestamp(f"2026-10-0{d} 10:00:00")
            rows += [dict(a=a, b=f"899{d}{k}{j:05d}", ts=t0 + pd.Timedelta(seconds=10 * j), dur=5)
                     for j in range(320)]
        for i in range(30):
            a = f"21000000{i:02d}"
            for j in range(12):
                b = f"81000{i:02d}{j:03d}"
                t = pd.Timestamp(f"2026-10-0{d} 09:00:00") + pd.Timedelta(minutes=40 * j + i)
                rows.append(dict(a=a, b=b, ts=t, dur=90))
                if j % 2 == 0:
                    rows.append(dict(a=b, b=a, ts=t + pd.Timedelta(minutes=5), dur=60))
    calls = pd.DataFrame(rows).assign(ctype="02", imsi=None, c=None)
    calls["dur"] = calls["dur"].astype("Int32")
    calls["ctype"] = calls["ctype"].astype(pd.CategoricalDtype(["01", "02", "03"]))
    res = model.temporal_check(calls)
    full = res["full"]
    assert full["positives"] == 4 and full["positives_new_numbers"] == 4
    assert full["recall"] == 1.0 and full["fp"] == 0


def test_run_end_to_end(data, tmp_path):
    f, s = data
    f.to_parquet(tmp_path / "features.parquet")
    s.to_parquet(tmp_path / "scores.parquet")
    report = model.run(tmp_path, tmp_path / "ml")
    assert report["agreement_oof"]["both"] == len(POS)
    ml = pd.read_parquet(tmp_path / "ml_scores.parquet")
    assert ml.loc[POS, "ml_proba"].min() >= model.THRESHOLD
    assert ml.loc[POS, "shap_factors"].notna().all()
    for name in ["metrics.json", "disagreements.csv", "anomalies.csv"]:
        assert (tmp_path / "ml" / name).exists()
    assert (tmp_path / "catboost_full.cbm").exists()
