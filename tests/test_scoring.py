import pandas as pd
import pytest

from src.scoring import score, write_answer

BASE = dict(n_out=6, uniq_b=6, repeat_share=0.0, peak_5m=1, peak_1h=2, share_short=0.1,
            callback_share=0.5, active_days=3, share_night=0.0, share_weekend=0.33,
            n_imsi=1, numbers_per_imsi_max=1, dur_median=60.0, operator="Альфа")

PROFILES = {
    "0252432858": dict(n_out=2231, uniq_b=2231, peak_5m=198, peak_1h=2231, share_short=0.49,
                       callback_share=0.0, active_days=1, share_weekend=0.0, dur_median=11.0),
    "2005272549": dict(n_out=750, uniq_b=750, peak_5m=40, peak_1h=412, share_short=1.0,
                       callback_share=0.03, active_days=1, share_weekend=0.0, dur_median=2.0),
    "2149482030": dict(n_out=918, uniq_b=918, peak_5m=58, peak_1h=250, share_short=0.46,
                       callback_share=0.0, active_days=1, share_weekend=0.0, dur_median=11.0),
    "0251594011": dict(n_out=441, uniq_b=441, peak_5m=30, peak_1h=200, share_short=1.0,
                       callback_share=0.02, active_days=1, share_weekend=0.0, dur_median=2.0),
    "4990496967": dict(n_out=375, uniq_b=375, peak_5m=20, peak_1h=375, share_short=0.1,
                       callback_share=0.0, active_days=1, share_weekend=0.0, operator="Городской"),
    "2145767275": dict(n_out=3936, uniq_b=3898, repeat_share=0.01, peak_5m=17, peak_1h=79,
                       share_short=0.11, callback_share=0.0, active_days=5, share_night=0.25,
                       share_weekend=0.35, operator="Bell-ka"),
    "4993831971": dict(n_out=1835, uniq_b=1829, repeat_share=0.003, peak_5m=15, peak_1h=87,
                       share_short=0.35, callback_share=0.0, active_days=3, share_weekend=0.0,
                       operator="Городской"),
    "8991234567": dict(n_imsi=6, numbers_per_imsi_max=1),
    "0261234567": dict(),
}


@pytest.fixture
def scores():
    rows = {n: {**BASE, **p} for n, p in PROFILES.items()}
    return score(pd.DataFrame.from_dict(rows, orient="index"))


def test_burst_is_high_risk(scores):
    s = scores.loc["0252432858"]
    assert s["level"] == "высокий" and s["pattern"] == "burst" and s["risk"] == 100
    assert s["reasons"].startswith("+40 2231 звонков за час")


def test_call_center_is_not_flagged(scores):
    s = scores.loc["4993831971"]
    assert s["level"] == "низкий" and s["legit_signs"]
    assert "-45 график офиса" in s["reasons"]


def test_round_clock_goes_to_review(scores):
    s = scores.loc["2145767275"]
    assert s["level"] == "проверка" and s["pattern"] == "round_clock"


def test_multi_sim_is_high(scores):
    s = scores.loc["8991234567"]
    assert s["level"] == "высокий" and s["pattern"] == "multi_sim"


def test_wangiri_is_high_despite_callbacks(scores):
    s = scores.loc["2005272549"]
    assert s["level"] == "высокий" and s["pattern"] == "wangiri"
    assert "перезванивают" not in s["reasons"]


def test_slow_wangiri_is_high(scores):
    s = scores.loc["0251594011"]
    assert s["level"] == "высокий" and s["pattern"] == "wangiri"


def test_slow_robot_is_high(scores):
    s = scores.loc["2149482030"]
    assert s["level"] == "высокий" and "профиль робота" in s["reasons"]


def test_city_burst_with_normal_calls_stays_in_review(scores):
    assert scores.loc["4990496967", "level"] == "проверка"


def test_ordinary_subscriber(scores):
    s = scores.loc["0261234567"]
    assert s["risk"] == 0 and s["reasons"] == ""


def test_answer_file(scores, tmp_path):
    path = tmp_path / "answer.txt"
    write_answer(scores, path)
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    assert sorted(lines) == ["0251594011;wangiri", "0252432858;burst", "2005272549;wangiri",
                             "2149482030;mass", "8991234567;multi_sim"]

    write_answer(scores, path, include_review=True)
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    assert {l.split(";")[0] for l in lines} == {"0251594011", "0252432858", "2005272549", "2149482030",
                                                "8991234567", "2145767275", "4990496967"}


def test_run_end_to_end(tmp_path):
    from src.scoring import run
    rows = {n: {**BASE, **p} for n, p in PROFILES.items()}
    feats = pd.DataFrame.from_dict(rows, orient="index")
    feats["operator"] = feats["operator"].astype("category")
    feats.to_parquet(tmp_path / "features.parquet")
    run(tmp_path, tmp_path / "answer.txt", tmp_path / "reports")
    assert (tmp_path / "scores.parquet").exists()
    assert len(pd.read_csv(tmp_path / "reports" / "candidates.csv", index_col=0, dtype={0: str})) == 8
