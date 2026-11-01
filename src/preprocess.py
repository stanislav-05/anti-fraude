from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.policy import EMERGENCY_NUMBERS


COLS = ["ctype", "imsi", "a", "b", "ts", "dur", "c"]
CALL_TYPES = ["01", "02", "03"]
OPERATORS = {
    "842": "Городской", "499": "Городской", "892": "Городской",
    "025": "Альфа", "026": "Альфа", "027": "Альфа",
    "899": "Барбетта", "810": "Барбетта", "834": "Барбетта",
    "210": "Bell-ka", "214": "Bell-ka", "200": "Bell-ka",
}
DATE_MIN = pd.Timestamp("2026-09-01")
DATE_MAX = pd.Timestamp("2026-11-01")
MAX_DURATION = 86_400

IMSI_PATTERN = r"250\d{12}"


def frame_from_lines(lines: list[str], src_file: str = "test.cdr") -> pd.DataFrame:
    """Список строк файла -> DataFrame (src_file, line_no, raw). Пустые строки пропускаем."""

    df = pd.DataFrame({
        "src_file": src_file,
        "line_no": np.arange(1, len(lines) + 1),
        "raw": pd.Series(lines, dtype="str"),
    })
    return df[df["raw"].str.strip() != ""]


def read_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Читает все *.cdr из папки. Каждая строка файла целиком попадает в колонку raw.

    Почему не pd.read_csv(sep=";")? В части строк лишние или недостающие поля,
    read_csv на них падает или сдвигает колонки. Надёжнее прочитать строку целиком
    и разбить её самим (split_fields).
    """

    files = sorted(Path(raw_dir).glob("*.cdr"))
    if not files:
        raise FileNotFoundError(f"В папке {raw_dir} нет файлов *.cdr")
    parts = [
        frame_from_lines(p.read_text(encoding="utf-8", errors="replace").splitlines(), p.name)
        for p in files
    ]
    return pd.concat(parts, ignore_index=True)


def split_fields(raw: pd.DataFrame) -> pd.DataFrame:
    """Разбивает raw по ';' на 7 полей и чинит строки с неверным числом полей.

    - больше 7 полей: берём первые 7, флаг fix_extra_fields;
    - 6 полей: одно пропущено. У переадресации (03) последним обычно стоит номер C,
      значит пропала длительность; у остальных типов пропал C. Флаг fix_missing_field;
    - меньше 6: восстановить нельзя, это решает clean() (карантин too_few_fields).
    """

    parts = raw["raw"].str.split(";", expand=True)
    parts = parts.reindex(columns=range(max(7, parts.shape[1])))
    n_fields = raw["raw"].str.count(";") + 1
    df = parts.iloc[:, :7].set_axis(COLS, axis=1).astype("str").fillna("")
    df = df.apply(lambda col: col.str.strip())
    looks_like_number = df["dur"].str.replace(r"\D", "", regex=True).str.len().ge(10)
    shift = n_fields.eq(6) & df["ctype"].eq("03") & looks_like_number
    df.loc[shift, "c"] = df.loc[shift, "dur"]
    df.loc[shift, "dur"] = ""
    df.insert(0, "src_file", raw["src_file"])
    df.insert(1, "line_no", raw["line_no"])
    df["raw"] = raw["raw"]
    df["n_fields"] = n_fields
    df["fix_extra_fields"] = n_fields.gt(7)
    df["fix_missing_field"] = n_fields.eq(6)
    return df


def normalize_number(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Номер -> ровно 10 цифр строкой (или NA) и флаг «номер пришлось исправить».

    Убираем всё, кроме цифр (+, пробелы, дефисы), срезаем код страны 7/8 у 11-значных.
    Номер остаётся строкой: у Альфы коды начинаются с нуля (0261234567).
    """

    digits = s.str.replace(r"\D", "", regex=True)
    has_country = digits.str.len().eq(11) & digits.str[0].isin(["7", "8"])
    digits = digits.mask(has_country, digits.str[1:])
    ok = digits.str.len().eq(10)
    result = digits.where(ok)
    fixed = ok & result.ne(s)
    return result, fixed.astype(bool)


def parse_timestamp(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Строка -> datetime (или NaT) и флаг «был нестандартный формат ГГГГ-ММ-ДД ЧЧ:ММ:СС»."""

    main = pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ts = main.fillna(iso)
    fixed_format = main.isna() & iso.notna()
    ts = ts.where(ts.between(DATE_MIN, DATE_MAX))
    return ts, fixed_format & ts.notna()


def parse_duration(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Длительность -> целые секунды (Int32, может быть NA) и два флага.

    Длительность не критична: если она битая, звонок всё равно был, и он нужен для
    подсчёта частоты. Поэтому строку не выбрасываем, а ставим NA.
    """

    num = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
    present = s.ne("")
    bad = present & (num.isna() | num.lt(0) | num.gt(MAX_DURATION))
    rounded = present & ~bad & num.ne(num.round())
    dur = num.round().where(present & ~bad).astype("Int32")
    return dur, bad.astype(bool), rounded.astype(bool)


def fix_call_type(ctype: pd.Series, imsi: pd.Series, c: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Тип вызова -> '01'/'02'/'03' и два флага.

    '1','2','3' -> потерян ведущий ноль, восстанавливаем.
    Мусор (AA, 00, 07…) -> выводим по структуре записи: есть номер C -> переадресация,
    есть IMSI -> исходящий, иначе входящий. Насколько это правило верно, проверяет
    type_rule_accuracy в отчёте качества.
    """

    zero_lost = ctype.isin(["1", "2", "3"])
    ctype = ctype.mask(zero_lost, "0" + ctype)
    unknown = ~ctype.isin(CALL_TYPES)
    ctype = ctype.mask(unknown, infer_call_type(imsi, c))
    return ctype, zero_lost, unknown


def infer_call_type(imsi: pd.Series, c: pd.Series) -> pd.Series:
    guess = np.select([c.notna(), imsi.notna()], ["03", "02"], default="01")
    return pd.Series(guess, index=imsi.index, dtype="str")


def clean(raw: pd.DataFrame, impute_dates: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """raw (src_file, line_no, raw) -> (calls, quarantine, stats).

    impute_dates=True: битая дата заменяется последней валидной датой из того же файла.
    Это корректно, только если записи в файле идут по времени, поэтому смотрите
    files_sorted_share в отчёте качества.
    """

    df = split_fields(raw)
    stats: dict[str, float] = {"rows_read": len(df), "files": df["src_file"].nunique()}
    a, fix_a = normalize_number(df["a"])
    b, fix_b = normalize_number(df["b"])
    b_digits = df["b"].str.replace(r"\D", "", regex=True)
    b_emergency = b.isna() & b_digits.isin(list(EMERGENCY_NUMBERS))
    b = b.mask(b_emergency, b_digits)
    c, fix_c = normalize_number(df["c"])
    bad_c = df["c"].ne("") & c.isna()
    imsi_ok = df["imsi"].str.fullmatch(IMSI_PATTERN).fillna(False).astype(bool)
    imsi = df["imsi"].where(imsi_ok)
    bad_imsi = df["imsi"].ne("") & ~imsi_ok
    ctype, type_zero, type_inferred = fix_call_type(df["ctype"], imsi, c)
    ts, date_format = parse_timestamp(df["ts"])
    dur, bad_dur, dur_rounded = parse_duration(df["dur"])
    reason = pd.Series(np.select(
        [df["n_fields"].lt(6), a.isna(), b.isna(), a.eq(b).fillna(False)],
        ["too_few_fields", "bad_number_a", "bad_number_b", "self_call"],
        default="",
    ), index=df.index)
    usable = reason.eq("")
    date_imputed = pd.Series(False, index=df.index)
    if impute_dates:
        filled = ts.where(usable).groupby(df["src_file"]).ffill()
        date_imputed = usable & ts.isna() & filled.notna()
        ts = ts.mask(date_imputed, filled)
    reason = reason.mask(usable & ts.isna(), "bad_date")
    stats["files_sorted_share"] = _files_sorted_share(df["src_file"], ts.where(~date_imputed))
    codes = pd.Series(list(OPERATORS))
    unknown_code = (
        ~a.str[:3].isin(codes) & a.notna()
        | ~b.str[:3].isin(codes) & b.notna() & ~b_emergency
        | ~c.str[:3].isin(codes) & c.notna()
    )
    out = pd.DataFrame({
        "src_file": df["src_file"], "line_no": df["line_no"],
        "ctype": ctype, "imsi": imsi, "a": a, "b": b, "ts": ts, "dur": dur, "c": c,
        "fix_extra_fields": df["fix_extra_fields"],
        "fix_missing_field": df["fix_missing_field"],
        "fix_country_code": fix_a | fix_b | fix_c,
        "fix_date_format": date_format,
        "fix_date_imputed": date_imputed,
        "fix_type_zero": type_zero,
        "fix_type_inferred": type_inferred,
        "fix_bad_duration": bad_dur,
        "fix_duration_rounded": dur_rounded,
        "fix_bad_imsi": bad_imsi,
        "fix_bad_c": bad_c,
        "warn_unknown_code": unknown_code,
        "info_emergency_b": b_emergency,
    })
    in_quarantine = reason.ne("")
    quarantine = df.loc[in_quarantine, ["src_file", "line_no", "raw"]].assign(reason=reason[in_quarantine])
    calls = out[~in_quarantine]
    dup = calls.duplicated(COLS, keep="first")
    stats["duplicates_removed"] = int(dup.sum())
    calls = calls[~dup].reset_index(drop=True)
    known = calls[~calls["fix_type_inferred"]]
    stats["type_rule_accuracy"] = float(
        infer_call_type(known["imsi"], known["c"]).eq(known["ctype"]).mean()
    ) if len(known) else float("nan")
    stats["rows_clean"] = len(calls)
    stats["rows_quarantine"] = len(quarantine)
    return calls, quarantine.reset_index(drop=True), stats


def _files_sorted_share(src_file: pd.Series, ts: pd.Series) -> float:
    """Доля файлов, где валидные даты идут по неубыванию (оправдывает ffill дат)."""

    per_file = ts.dropna().groupby(src_file).apply(lambda s: s.is_monotonic_increasing)
    return float(per_file.mean()) if len(per_file) else float("nan")


def add_derived(calls: pd.DataFrame) -> pd.DataFrame:
    """Добавляет поля для EDA и признаков. Возвращает новый DataFrame."""

    out = calls.copy()
    out["day"] = out["ts"].dt.normalize()
    out["hour"] = out["ts"].dt.hour.astype("int8")
    out["weekday"] = out["ts"].dt.day_of_week.astype("int8")
    out["day_part"] = pd.cut(out["hour"], bins=[-1, 5, 11, 17, 23],
                             labels=["ночь", "утро", "день", "вечер"])
    out["dur_cat"] = pd.cut(out["dur"], bins=[-1, 0, 2, 10, 60, 600, MAX_DURATION],
                            labels=["0", "1-2 с", "3-10 с", "11-60 с", "1-10 мин", ">10 мин"])
    for col in ["a", "b", "c"]:
        out[f"op_{col}"] = out[col].str[:3].map(OPERATORS).astype("category")
    out["op_a"] = out["op_a"].cat.add_categories("неизвестный").fillna("неизвестный")
    out["op_b"] = out["op_b"].cat.add_categories("неизвестный").fillna("неизвестный")
    out["ctype"] = out["ctype"].astype(pd.CategoricalDtype(CALL_TYPES))
    out["is_forward"] = out["ctype"].eq("03")
    return out


def _fmt_value(v: float) -> str:
    """Целые — с разделителями тысяч, доли — 6 значащих цифр."""

    return f"{v:,.0f}" if float(v).is_integer() else f"{v:.6g}"


def quality_report(calls: pd.DataFrame, quarantine: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Сводная таблица: сколько строк исправлено / отброшено / какие метрики."""

    fixes = calls.filter(regex=r"^(fix|warn|info)_").sum()
    rows = [(k.split("_", 1)[0], k, int(v)) for k, v in fixes.items()]
    rows += [("quarantine", k, int(v)) for k, v in quarantine["reason"].value_counts().items()]
    rows += [("metric", k, v) for k, v in stats.items()]
    return pd.DataFrame(rows, columns=["kind", "name", "value"])


def run(raw_dir: str | Path = "data/raw", out_dir: str | Path = "data/processed",
        impute_dates: bool = True) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw(raw_dir)
    calls, quarantine, stats = clean(raw, impute_dates=impute_dates)
    calls = add_derived(calls)
    calls.to_parquet(out_dir / "calls.parquet", index=False)
    quarantine.to_parquet(out_dir / "quarantine.parquet", index=False)
    report = quality_report(calls, quarantine, stats)
    report.to_csv(out_dir / "quality.csv", index=False)
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(report.to_string(index=False, formatters={"value": _fmt_value}))
    print(f"\nСохранено в {out_dir}: calls {len(calls):,}, карантин {len(quarantine):,}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Очистка CDR -> parquet")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--no-impute-dates", action="store_true",
                    help="не восстанавливать битые даты, а отправлять строки в карантин")
    args = ap.parse_args()
    run(args.raw, args.out, impute_dates=not args.no_impute_dates)
