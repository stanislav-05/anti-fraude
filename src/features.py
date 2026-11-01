from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import OPERATORS

CITY_CODES = ["842", "499", "892"]
SHORT_CALL = 10
LONG_CALL = 120
WINDOWS = {"5m": 5 * 60, "15m": 15 * 60, "1h": 60 * 60, "24h": 24 * 60 * 60}

FEATURES = {
    "n_out": "Исходящих вызовов за весь период",
    "uniq_b": "Разных получателей",
    "repeat_share": "Доля повторных звонков тем же получателям (1 − uniq_b / n_out)",
    "dur_mean": "Средняя длительность исходящих, сек",
    "dur_median": "Медианная длительность исходящих, сек",
    "dur_p90": "90-й перцентиль длительности, сек",
    "share_dur0": "Доля исходящих с длительностью 0 (не равно «неотвеченный»: статуса ответа нет)",
    "share_le2": "Доля исходящих до 2 сек — сброс на первых секундах",
    "share_short": f"Доля исходящих до {SHORT_CALL} сек",
    "share_long": f"Доля исходящих длиннее {LONG_CALL} сек",
    "share_dur_missing": "Доля исходящих с неизвестной (битой) длительностью",
    **{f"peak_{w}": f"Максимум исходящих в скользящем окне {w}" for w in WINDOWS},
    "max_calls_day": "Максимум исходящих за календарный день",
    "active_days": "Дней с исходящими",
    "active_hour_slots": "Разных часов (дата+час) с исходящими",
    "calls_per_active_hour": "Исходящих в среднем на активный час",
    "span_hours": "Часов между первым и последним исходящим",
    "gap_median": "Медианная пауза между началами соседних исходящих, сек",
    "share_gap_le60": "Доля пауз между исходящими не больше минуты («конвейер»)",
    "share_night": "Доля исходящих ночью (00–06)",
    "share_workhours": "Доля исходящих в рабочие часы (09–18)",
    "share_weekend": "Доля исходящих в субботу и воскресенье (3–4 октября 2026)",
    "hour_entropy": "Равномерность по часам суток: 0 — все в один час, 1 — равномерно",
    "n_b_codes": "Разных кодов (оператор/регион) у получателей",
    "n_b_operators": "Разных операторов у получателей",
    "share_b_city": "Доля исходящих на городские номера",
    "share_forward": "Доля исходящих, которые B переадресовал дальше (тип 03)",
    "callback_share": "Доля получателей, которые сами звонили этому номеру",
    "n_imsi": "Разных IMSI у номера",
    "numbers_per_imsi_max": "Сколько номеров максимум делят одну SIM (IMSI) с этим номером",
    "n_in": "Входящих вызовов",
    "uniq_a": "Разных звонивших",
    "share_in_short": f"Доля входящих до {SHORT_CALL} сек",
    "in_out_ratio": "Входящих на один исходящий: n_in / (n_out + 1)",
    "n_forwarded": "Сколько входящих номер переадресовал (B в записях 03)",
    "n_as_forward_target": "Сколько раз на номер переадресовывали (C в записях 03)",
    "operator": "Оператор по коду номера",
    "is_city": "Городской номер",
    "unknown_code": "Код оператора не из справочника",
}


def window_counts(group_id: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    """Для каждой строки: сколько звонков того же номера начались в [t, t + window).

    Данные должны быть отсортированы по (group_id, t). Трюк: склеиваем номер и время
    в один возрастающий ключ group_id * BIG + t, где BIG больше любого разброса времени.
    Тогда np.searchsorted за один вызов находит конец окна для всех строк сразу.
    Максимум этой величины по номеру = пик звонков в любом окне длины window.
    """

    t = t - t.min() if len(t) else t
    big = (t.max() if len(t) else 0) + window + 1
    key = group_id.astype(np.int64) * big + t
    end = np.searchsorted(key, key + window, side="left")
    return end - np.arange(len(key))


def outgoing_features(calls: pd.DataFrame) -> pd.DataFrame:
    """Признаки номера как инициатора вызова (A)."""

    o = calls[["a", "b", "ts", "dur", "ctype"]].sort_values(["a", "ts"], kind="stable")
    o = o.reset_index(drop=True)
    o["t"] = (o["ts"] - o["ts"].min()).dt.total_seconds().astype("int64")
    o["hour"] = o["ts"].dt.hour
    o["day"] = o["ts"].dt.normalize()
    o["hour_slot"] = o["ts"].dt.floor("h")
    dur = o["dur"].astype("float64")
    known = dur.notna()
    o["dur_f"] = dur
    o["is_zero"] = dur.eq(0).astype(float).where(known)
    o["is_le2"] = dur.le(2).astype(float).where(known)
    o["is_short"] = dur.le(SHORT_CALL).astype(float).where(known)
    o["is_long"] = dur.gt(LONG_CALL).astype(float).where(known)
    o["dur_missing"] = (~known).astype(float)
    o["night"] = o["hour"].lt(6).astype(float)
    o["workh"] = o["hour"].between(9, 17).astype(float)
    o["weekend"] = o["ts"].dt.day_of_week.ge(5).astype(float)
    o["b_code"] = o["b"].str[:3]
    o["b_operator"] = o["b_code"].map(OPERATORS)
    o["b_city"] = o["b_code"].isin(CITY_CODES).astype(float)
    o["is_forward"] = o["ctype"].astype(str).eq("03").astype(float)

    gap = o.groupby("a", sort=False)["t"].diff()
    o["gap"] = gap
    o["gap_le60"] = gap.le(60).astype(float).where(gap.notna())
    gid = pd.factorize(o["a"])[0]
    for name, seconds in WINDOWS.items():
        o[f"c_{name}"] = window_counts(gid, o["t"].to_numpy(), seconds)

    g = o.groupby("a", sort=False)
    f = pd.DataFrame({
        "n_out": g.size(),
        "uniq_b": g["b"].nunique(),
        "dur_mean": g["dur_f"].mean(),
        "dur_median": g["dur_f"].median(),
        "dur_p90": g["dur_f"].quantile(0.9),
        "share_dur0": g["is_zero"].mean(),
        "share_le2": g["is_le2"].mean(),
        "share_short": g["is_short"].mean(),
        "share_long": g["is_long"].mean(),
        "share_dur_missing": g["dur_missing"].mean(),
        **{f"peak_{w}": g[f"c_{w}"].max() for w in WINDOWS},
        "active_days": g["day"].nunique(),
        "active_hour_slots": g["hour_slot"].nunique(),
        "span_hours": (g["t"].max() - g["t"].min()) / 3600,
        "gap_median": g["gap"].median(),
        "share_gap_le60": g["gap_le60"].mean(),
        "share_night": g["night"].mean(),
        "share_workhours": g["workh"].mean(),
        "share_weekend": g["weekend"].mean(),
        "n_b_codes": g["b_code"].nunique(),
        "n_b_operators": g["b_operator"].nunique(),
        "share_b_city": g["b_city"].mean(),
        "share_forward": g["is_forward"].mean(),
    })
    f["repeat_share"] = 1 - f["uniq_b"] / f["n_out"]
    f["calls_per_active_hour"] = f["n_out"] / f["active_hour_slots"]
    f["max_calls_day"] = o.groupby(["a", "day"]).size().groupby(level=0).max()
    f["hour_entropy"] = hour_entropy(o)
    f["callback_share"] = callback_share(calls)
    f.index.name = "number"
    return f


def hour_entropy(o: pd.DataFrame) -> pd.Series:
    """Энтропия распределения звонков по 24 часам, нормированная в [0, 1]."""
    
    per_hour = o.groupby(["a", "hour"]).size()
    p = per_hour / per_hour.groupby(level=0).transform("sum")
    return (-(p * np.log(p)).groupby(level=0).sum() / np.log(24)).clip(lower=0)


def callback_share(calls: pd.DataFrame) -> pd.Series:
    """Для каждого A: доля его получателей B, которые тоже звонили A.

    Обычные люди перезванивают друг другу; номер, который обзванивает базу,
    обратных звонков почти не получает.
    """

    pairs = calls[["a", "b"]].drop_duplicates()
    forward = pd.MultiIndex.from_frame(pairs)
    backward = pd.MultiIndex.from_arrays([pairs["b"], pairs["a"]])
    pairs = pairs.assign(back=backward.isin(forward).astype(float))
    return pairs.groupby("a")["back"].mean()


def sim_features(calls: pd.DataFrame) -> pd.DataFrame:
    """Связь номер ↔ SIM: сколько IMSI у номера и сколько номеров на одной SIM."""

    pairs = calls.loc[calls["imsi"].notna(), ["a", "imsi"]].drop_duplicates()
    per_imsi = pairs.groupby("imsi")["a"].nunique()
    pairs = pairs.assign(numbers_on_sim=pairs["imsi"].map(per_imsi))
    g = pairs.groupby("a")
    return pd.DataFrame({
        "n_imsi": g["imsi"].nunique(),
        "numbers_per_imsi_max": g["numbers_on_sim"].max(),
    })


def incoming_features(calls: pd.DataFrame) -> pd.DataFrame:
    """Признаки номера как получателя (B) и участника переадресации."""

    i = calls[["a", "b", "dur", "ctype", "c"]]
    short = i["dur"].astype("float64").le(SHORT_CALL).astype(float).where(i["dur"].notna())
    g = i.assign(short=short).groupby("b")
    f = pd.DataFrame({
        "n_in": g.size(),
        "uniq_a": g["a"].nunique(),
        "share_in_short": g["short"].mean(),
    })
    fwd = i[i["ctype"].astype(str).eq("03")]
    f = f.join(fwd.groupby("b").size().rename("n_forwarded"), how="outer")
    f = f.join(fwd.dropna(subset=["c"]).groupby("c").size().rename("n_as_forward_target"), how="outer")
    return f


COUNT_COLUMNS = ["n_out", "uniq_b", "n_in", "uniq_a", "n_forwarded", "n_as_forward_target",
                 "n_imsi", "active_days", "active_hour_slots", "max_calls_day",
                 *[f"peak_{w}" for w in WINDOWS]]


def build_features(calls: pd.DataFrame) -> pd.DataFrame:
    """calls (результат src.preprocess) -> признаки, индекс = номер."""

    out = outgoing_features(calls)
    feats = out.join(sim_features(calls), how="left")
    feats = feats.join(incoming_features(calls), how="outer")
    feats.index.name = "number"
    feats[COUNT_COLUMNS] = feats[COUNT_COLUMNS].fillna(0).astype("int64")
    feats["in_out_ratio"] = feats["n_in"] / (feats["n_out"] + 1)
    code = feats.index.str[:3]
    feats["operator"] = pd.Series(code, index=feats.index).map(OPERATORS).fillna("неизвестный").astype("category")
    feats["is_city"] = code.isin(CITY_CODES)
    feats["unknown_code"] = ~code.isin(list(OPERATORS))
    return feats[[c for c in FEATURES if c in feats.columns]].sort_index()


def run(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    calls = pd.read_parquet(processed_dir / "calls.parquet",
                            columns=["ctype", "imsi", "a", "b", "ts", "dur", "c"])
    feats = build_features(calls)
    feats.to_parquet(processed_dir / "features.parquet")
    pd.Series(FEATURES, name="description").rename_axis("feature").to_csv(
        processed_dir / "features_description.csv")
    
    callers = feats[feats["n_out"] > 0]
    print(f"Номеров всего: {len(feats):,}; звонили сами: {len(callers):,}")
    cols = ["n_out", "uniq_b", "peak_5m", "peak_1h", "dur_median", "share_short", "callback_share", "operator"]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nТоп-15 по числу исходящих:")
        print(callers.nlargest(15, "n_out")[cols].round(2).to_string())
    print(f"\nСохранено: {processed_dir / 'features.parquet'}")
    return feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Признаки по номерам")
    ap.add_argument("--processed", default="data/processed")
    run(ap.parse_args().processed)
