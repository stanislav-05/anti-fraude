import pandas as pd
import pytest

from src.preprocess import add_derived, clean, frame_from_lines, read_raw, run

GOOD = "02;250021234567890;8991234567;0261234567;20261001083015;54;"


def run_clean(*lines, **kw):
    calls, quarantine, stats = clean(frame_from_lines(list(lines)), **kw)
    return calls, quarantine, stats


def one_row(line):
    calls, quarantine, _ = run_clean(line)
    assert len(calls) == 1, f"строка ушла в карантин: {quarantine.to_dict('records')}"
    return calls.iloc[0]


def test_good_line_has_no_flags():
    row = one_row(GOOD)
    assert row["ctype"] == "02"
    assert row["a"] == "8991234567"
    assert row["b"] == "0261234567"
    assert row["ts"] == pd.Timestamp("2026-10-01 08:30:15")
    assert row["dur"] == 54
    assert pd.isna(row["c"])
    flags = row.filter(regex=r"^(fix|warn)_")
    assert not flags.any(), flags[flags]


def test_forward_call_keeps_number_c():
    row = one_row("03;;8421234567;2101234567;20261001100000;12;2141234567")
    assert row["ctype"] == "03" and row["c"] == "2141234567"


@pytest.mark.parametrize("raw_a", ["88991234567", "78991234567", "+7 899 123-45-67"])
def test_country_code_is_removed(raw_a):
    row = one_row(f"02;250021234567890;{raw_a};0261234567;20261001083015;54;")
    assert row["a"] == "8991234567"
    assert row["fix_country_code"]


@pytest.mark.parametrize("line, reason", [
    ("02;;12345;0261234567;20261001083015;54;", "bad_number_a"),
    ("02;;8991234567;;20261001083015;54;", "bad_number_b"),
    ("02;;8991234567;8991234567;20261001083015;54;", "self_call"),
    ("02;;8991234567;0261234567", "too_few_fields"),
])
def test_unrecoverable_rows_go_to_quarantine(line, reason):
    calls, quarantine, _ = run_clean(line)
    assert calls.empty
    assert quarantine["reason"].tolist() == [reason]
    assert quarantine["raw"].iloc[0] == line


def test_unknown_operator_code_is_kept_but_marked():
    row = one_row("01;;7771234567;0261234567;20261001083015;5;")
    assert row["a"] == "7771234567" and row["warn_unknown_code"]


def test_extra_fields_are_dropped():
    row = one_row(GOOD + ";;мусор")
    assert row["fix_extra_fields"] and row["dur"] == 54


def test_missing_duration_in_forward_call():
    row = one_row("03;;8421234567;2101234567;20261001100000;2141234567")
    assert row["c"] == "2141234567" and pd.isna(row["dur"])
    assert row["fix_missing_field"]


def test_missing_last_field_in_regular_call():
    row = one_row("02;250021234567890;8991234567;0261234567;20261001083015;54")
    assert row["dur"] == 54 and pd.isna(row["c"]) and row["fix_missing_field"]


def test_iso_date_format_is_parsed():
    row = one_row("02;;8991234567;0261234567;2026-10-01 08:30:15;54;")
    assert row["ts"] == pd.Timestamp("2026-10-01 08:30:15") and row["fix_date_format"]


def test_bad_date_is_imputed_from_previous_line():
    calls, _, _ = run_clean(GOOD, "02;;8101234567;0271234567;20261399999999;10;")
    assert calls["ts"].iloc[1] == pd.Timestamp("2026-10-01 08:30:15")
    assert calls["fix_date_imputed"].tolist() == [False, True]


def test_bad_date_without_previous_goes_to_quarantine():
    _, quarantine, _ = run_clean("02;;8101234567;0271234567;19700101000000;10;")
    assert quarantine["reason"].tolist() == ["bad_date"]


def test_imputation_can_be_turned_off():
    _, quarantine, _ = run_clean(GOOD, "02;;8101234567;0271234567;oops;10;", impute_dates=False)
    assert quarantine["reason"].tolist() == ["bad_date"]


def test_lost_leading_zero_in_type():
    row = one_row("2;250021234567890;8991234567;0261234567;20261001083015;54;")
    assert row["ctype"] == "02" and row["fix_type_zero"]


@pytest.mark.parametrize("line, expected", [
    ("AA;250021234567890;8991234567;0261234567;20261001083015;54;", "02"),
    ("07;;8421234567;0261234567;20261001083015;54;", "01"),
    ("00;;8421234567;2101234567;20261001100000;12;2141234567", "03"),
])
def test_unknown_type_is_inferred(line, expected):
    row = one_row(line)
    assert row["ctype"] == expected and row["fix_type_inferred"]


@pytest.mark.parametrize("raw_dur", ["99999999", "-5", "abc"])
def test_bad_duration_becomes_na_but_row_is_kept(raw_dur):
    row = one_row(f"02;;8991234567;0261234567;20261001083015;{raw_dur};")
    assert pd.isna(row["dur"]) and row["fix_bad_duration"]


@pytest.mark.parametrize("raw_dur, expected", [("12.6", 13), ("7,2", 7)])
def test_fractional_duration_is_rounded(raw_dur, expected):
    row = one_row(f"02;;8991234567;0261234567;20261001083015;{raw_dur};")
    assert row["dur"] == expected and row["fix_duration_rounded"]


def test_truncated_imsi_becomes_na():
    row = one_row("02;25002123;8991234567;0261234567;20261001083015;54;")
    assert pd.isna(row["imsi"]) and row["fix_bad_imsi"]


def test_exact_duplicates_are_removed():
    calls, _, stats = run_clean(GOOD, GOOD, GOOD)
    assert len(calls) == 1 and stats["duplicates_removed"] == 2


def test_type_rule_accuracy_is_reported():
    _, _, stats = run_clean(GOOD, "01;;8421234567;0261234567;20261001083015;5;")
    assert stats["type_rule_accuracy"] == 1.0


def test_derived_fields():
    calls, _, _ = run_clean(GOOD, "01;;7771234567;0261234567;20261001233000;0;")
    d = add_derived(calls)
    assert d["hour"].tolist() == [8, 23]
    assert d["day_part"].astype(str).tolist() == ["утро", "вечер"]
    assert d["dur_cat"].astype(str).tolist() == ["11-60 с", "0"]
    assert d["op_a"].astype(str).tolist() == ["Барбетта", "неизвестный"]
    assert d["op_b"].astype(str).tolist() == ["Альфа", "Альфа"]


def test_read_raw_skips_blank_lines(tmp_path):
    (tmp_path / "CDR_1.cdr").write_text(GOOD + "\n\n" + GOOD + "\n", encoding="utf-8")
    (tmp_path / "CDR_2.cdr").write_text(GOOD + "\n", encoding="utf-8")
    raw = read_raw(tmp_path)
    assert len(raw) == 3
    assert raw["line_no"].tolist() == [1, 3, 1]


def test_read_raw_without_files_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_raw(tmp_path)


def test_full_run_writes_parquet_and_keeps_leading_zeros(tmp_path):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "CDR_1.cdr").write_text(
        "\n".join([GOOD, "02;;12345;0261234567;20261001083015;54;"]), encoding="utf-8")
    run(raw_dir, out_dir)

    calls = pd.read_parquet(out_dir / "calls.parquet")
    assert calls["b"].iloc[0] == "0261234567"
    assert len(pd.read_parquet(out_dir / "quarantine.parquet")) == 1
    report = pd.read_csv(out_dir / "quality.csv")
    assert {"fix", "quarantine", "metric"} <= set(report["kind"])
