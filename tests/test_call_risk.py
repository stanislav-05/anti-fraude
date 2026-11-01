import pandas as pd
import pytest

from src.call_risk import card, mask_number, pick_examples, run, score_calls


W = "8991111111"
B = "0252432858"
C = "4993831971"
N1, N2 = "2101111111", "2102222222"
V = "0261111111"
SCORES = pd.DataFrame({
    "risk": [100, 100, 0, 0, 0, 0],
    "level": ["высокий", "высокий", "низкий", "низкий", "низкий", "низкий"],
    "pattern": ["wangiri", "burst", "", "", "", ""],
    "reasons": ["+45 100% звонков короче 10 с — «звонок-сброс» (wangiri); +40 412 звонков за час",
                "+40 2231 звонков за час; +35 профиль робота: медиана 11 с; +15 198 звонков за 5 минут; "
                "+10 никто из 2231 получателей не перезвонил",
                "-45 график офиса: 3 будних дня", "", "", ""],
    "legit_signs": [False, False, True, False, False, False],
    "n_out": [750, 2231, 1835, 2, 1, 1],
}, index=[W, B, C, N1, N2, V])


def call(ctype, a, b, ts, dur):
    return {"ctype": ctype, "a": a, "b": b, "ts": pd.Timestamp(ts), "dur": dur}


CALLS = pd.DataFrame([
    call("02", W, V, "2026-10-03 03:00:00", 1),
    call("02", V, W, "2026-10-03 10:00:00", 40),
    call("02", N1, N2, "2026-10-02 14:00:00", 120),
    call("02", N2, N1, "2026-10-02 15:00:00", 60),
    call("02", N1, "112", "2026-10-04 12:00:00", 30),
    call("01", C, N1, "2026-10-01 11:00:00", 60),
    call("02", B, N2, "2026-10-02 16:43:57", 11),
]).astype({"dur": "Int32"})


@pytest.fixture(scope="module")
def scored():
    return score_calls(CALLS, SCORES)


def test_wangiri_drop_call_is_high(scored):
    r = scored.loc[0]
    assert r["risk"] == 100 and r["level"] == "высокий"
    assert r["call_points"] == 10 + 5


def test_victim_callback_is_flagged(scored):
    r = scored.loc[1]
    assert r["a_risk"] == 0 and r["risk"] == 70 and r["level"] == "высокий"


def test_ordinary_call_is_low(scored):
    assert scored.loc[2, "risk"] == 0 and scored.loc[2, "level"] == "низкий"


def test_emergency_call_is_never_scored(scored):
    r = scored.loc[4]
    assert r["whitelisted"] and r["risk"] == 0 and r["level"] == "низкий"


def test_no_autoblock_anywhere(scored):
    assert not scored["autoblock"].any()


def test_mask_number():
    assert mask_number("0252432858") == "+7 025 ***-**-58"
    assert mask_number("112") == "112"


def test_card_looks_like_the_spec(scored):
    text = card(scored, 0, SCORES)
    assert text.splitlines()[0] == "+7 899 ***-**-11 → +7 026 ***-**-11"
    assert "3 окт, 03:00:00 · 1 сек · исходящий" in text
    assert "РИСК 100" in text
    assert "+45  номер A: 100% звонков короче 10 с" in text
    assert "+10  сброс на первых секундах: 1 с" in text
    assert "риск ограничен" not in text


def test_card_caps_risk_at_100(scored):
    text = card(scored, 6, SCORES)
    assert "РИСК 100" in text and "сумма факторов 105, риск ограничен" in text
    assert text.endswith("автоблокировка: нет")


def test_card_for_emergency_call(scored):
    text = card(scored, 4, SCORES)
    assert "→ 112" in text and "белый список" in text and "РИСК 0" in text


def test_examples_cover_key_cases(scored):
    ex = pick_examples(scored, SCORES)
    assert ex["номер схемы wangiri"] == 0
    assert ex["жертва перезванивает на номер wangiri"] == 1
    assert ex["колл-центр (легальный массовый источник)"] == 5


def test_run_end_to_end(tmp_path):
    CALLS.to_parquet(tmp_path / "calls.parquet", index=False)
    SCORES.to_parquet(tmp_path / "scores.parquet")
    run(tmp_path, tmp_path / "reports")
    out = pd.read_parquet(tmp_path / "call_scores.parquet")
    assert len(out) == len(CALLS) and out["risk"].between(0, 100).all()
    assert "РИСК" in (tmp_path / "reports" / "call_cards.md").read_text(encoding="utf-8")
