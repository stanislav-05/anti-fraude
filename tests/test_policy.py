import pandas as pd

from src import policy
from src.preprocess import clean, frame_from_lines
from src.scoring import score, write_answer

AMBULANCE_DISPATCH = "8421234567"


def test_autoblock_is_off():
    assert policy.AUTOBLOCK is False


def test_emergency_numbers_are_whitelisted():
    for n in ["112", "103", "03"]:
        assert n in policy.EMERGENCY_NUMBERS


def test_whitelist_file_extends_emergency_list(tmp_path):
    f = tmp_path / "whitelist.txt"
    f.write_text(f"# диспетчерская скорой\n{AMBULANCE_DISPATCH}  # комментарий\n\n", encoding="utf-8")
    wl = policy.load_whitelist(f)
    assert AMBULANCE_DISPATCH in wl and "112" in wl
    assert policy.load_whitelist(tmp_path / "нет_файла.txt") == policy.EMERGENCY_NUMBERS


def test_call_to_112_is_not_lost_in_cleaning():
    calls, quarantine, _ = clean(frame_from_lines(["02;250021234567890;8991234567;112;20261001083015;54;"]))
    assert quarantine.empty
    row = calls.iloc[0]
    assert row["b"] == "112" and row["info_emergency_b"] and not row["warn_unknown_code"]


def test_dispatch_number_with_robot_profile_is_never_in_answer(tmp_path):
    robot_like = dict(n_out=2000, uniq_b=2000, repeat_share=0.0, peak_5m=200, peak_1h=1500,
                      share_short=0.9, callback_share=0.0, active_days=5, share_night=0.3,
                      share_weekend=0.4, n_imsi=1, numbers_per_imsi_max=1, dur_median=5.0,
                      operator="Городской")
    feats = pd.DataFrame.from_dict({AMBULANCE_DISPATCH: robot_like, "0252432858": robot_like},
                                   orient="index")

    without = score(feats)
    assert without.loc[AMBULANCE_DISPATCH, "level"] == "высокий"
    scores = score(feats, policy.load_whitelist(None) | {AMBULANCE_DISPATCH})
    r = scores.loc[AMBULANCE_DISPATCH]
    assert r["risk"] == 0 and r["whitelisted"] and "белый список" in r["reasons"]
    answer = write_answer(scores, tmp_path / "answer.txt", include_review=True)
    assert AMBULANCE_DISPATCH not in set(answer)
    assert "0252432858" in set(answer)
