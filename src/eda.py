from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["a", "b", "ts"]


def load_calls(path: str | Path = "data/processed/calls.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def overview(calls: pd.DataFrame) -> pd.Series:
    numbers = pd.concat([calls["a"], calls["b"], calls["c"].dropna()]).nunique()
    return pd.Series({
        "записей": len(calls),
        "период с": calls["ts"].min(),
        "период по": calls["ts"].max(),
        "уникальных номеров A": calls["a"].nunique(),
        "уникальных номеров B": calls["b"].nunique(),
        "уникальных номеров всего (A, B, C)": numbers,
        "уникальных IMSI": calls["imsi"].nunique(),
        "доля записей с IMSI": calls["imsi"].notna().mean(),
        "доля с неизвестной длительностью": calls["dur"].isna().mean(),
    })


def calls_by_day_hour(calls: pd.DataFrame) -> pd.DataFrame:
    """Количество звонков: строки — дни, столбцы — часы."""

    return calls.pivot_table(index="day", columns="hour", values="a", aggfunc="size", fill_value=0)


def type_semantics(calls: pd.DataFrame) -> pd.DataFrame:
    """Проверка смысла типов: у кого из типов есть IMSI и номер C.

    Ожидаем: у 03 всегда есть C, у остальных нет; IMSI есть у мобильных A.
    """

    t = calls.assign(has_imsi=calls["imsi"].notna(), has_c=calls["c"].notna(),
                     a_city=calls["op_a"].astype(str).eq("Городской"))
    return t.groupby("ctype", observed=True).agg(
        records=("a", "size"),
        share_with_imsi=("has_imsi", "mean"),
        share_with_c=("has_c", "mean"),
        share_a_city=("a_city", "mean"),
    )


def mirrored_records(calls: pd.DataFrame) -> pd.Series:
    """Не записан ли один звонок дважды (например, 02 у A и 01 у B)?

    Если такие пары есть, счётчики звонков будут завышены, и их надо схлопывать.
    """

    types_per_call = calls.groupby(KEY, observed=True)["ctype"].nunique()
    same_call = calls.groupby(KEY, observed=True).size()
    return pd.Series({
        "звонков (a, b, ts) встречается больше 1 раза": int((same_call > 1).sum()),
        "из них с разными типами записей": int((types_per_call > 1).sum()),
    })


def duration_summary(calls: pd.DataFrame) -> pd.Series:
    d = calls["dur"].astype("float64")
    return pd.concat([
        d.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]),
        pd.Series({"доля 0 сек": d.eq(0).mean(), "доля ≤ 2 сек": d.le(2).mean(),
                   "доля ≤ 10 сек": d.le(10).mean()}),
    ])


def duration_by_category(calls: pd.DataFrame) -> pd.DataFrame:
    """Категории длительности по типам вызова (доли внутри типа)."""

    return pd.crosstab(calls["dur_cat"], calls["ctype"], normalize="columns").round(4)


def operator_matrix(calls: pd.DataFrame) -> pd.DataFrame:
    """Кто кому звонит: оператор A × оператор B, число записей."""

    return pd.crosstab(calls["op_a"], calls["op_b"], margins=True, margins_name="всего")


def unknown_codes(calls: pd.DataFrame) -> pd.Series:
    """Коды номеров, которых нет в справочнике (warn_unknown_code)."""

    u = calls[calls["warn_unknown_code"]]
    from src.preprocess import OPERATORS
    codes = pd.concat([u["a"], u["b"], u["c"].dropna()]).str[:3]
    return codes[~codes.isin(list(OPERATORS))].value_counts()


def calls_per_number(calls: pd.DataFrame) -> pd.Series:
    """Сколько звонков инициировал каждый номер A."""

    return calls.groupby("a").size().rename("n_out")


def activity_distribution(calls: pd.DataFrame) -> pd.Series:
    n = calls_per_number(calls)
    return n.describe(percentiles=[0.5, 0.9, 0.99, 0.999])


def top_callers(calls: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Самые активные A и простые характеристики их звонков."""

    d = calls.assign(dur_f=calls["dur"].astype("float64"),
                     short=calls["dur"].astype("float64").le(10))
    g = d.groupby("a")
    top = pd.DataFrame({
        "n_out": g.size(),
        "uniq_b": g["b"].nunique(),
        "dur_median": g["dur_f"].median(),
        "share_short": g["short"].mean(),
        "active_days": g["day"].nunique(),
        "first": g["ts"].min(),
        "last": g["ts"].max(),
    }).nlargest(n, "n_out")
    top["operator"] = d.drop_duplicates("a").set_index("a").loc[top.index, "op_a"]
    return top


def daily_stability(calls: pd.DataFrame, numbers: list[str] | pd.Index) -> pd.DataFrame:
    """Звонки выбранных номеров по дням — устойчив ли паттерн с 1 по 5 октября."""

    sel = calls[calls["a"].isin(numbers)]
    return sel.pivot_table(index="a", columns="day", values="b", aggfunc="size", fill_value=0)


def sim_check(calls: pd.DataFrame) -> pd.DataFrame:
    """Сколько номеров на одной SIM и сколько SIM у одного номера."""

    pairs = calls.loc[calls["imsi"].notna(), ["a", "imsi"]].drop_duplicates()
    return pd.DataFrame({
        "номеров на IMSI": pairs.groupby("imsi")["a"].nunique().value_counts().sort_index(),
        "IMSI на номер": pairs.groupby("a")["imsi"].nunique().value_counts().sort_index(),
    }).fillna(0).astype(int).rename_axis("сколько")



def save_figures(calls: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    pivot = calls_by_day_hour(calls)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), [d.strftime("%d.%m") for d in pivot.index])
    ax.set_xlabel("час")
    ax.set_title("Звонки по дням и часам")
    fig.colorbar(im, ax=ax, label="звонков")
    saved.append(_save(fig, out_dir / "calls_day_hour.png"))
    d = calls["dur"].dropna().astype("int64")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(d.clip(upper=900), bins=np.arange(0, 905, 5), color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("длительность, сек (всё дольше 15 мин — в последнем столбце)")
    ax.set_ylabel("звонков (лог. шкала)")
    ax.set_title("Распределение длительности")
    saved.append(_save(fig, out_dir / "duration_hist.png"))
    n = calls_per_number(calls)
    counts = n.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.loglog(counts.index, counts.to_numpy(), "o", ms=3, color="steelblue")
    ax.set_xlabel("исходящих звонков у номера")
    ax.set_ylabel("номеров")
    ax.set_title("Активность номеров (лог-лог): хвост справа — кандидаты в массовые обзвоны")
    saved.append(_save(fig, out_dir / "calls_per_number.png"))
    return saved


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def run(calls_path: str | Path = "data/processed/calls.parquet",
        out_dir: str | Path = "reports/eda") -> dict[str, pd.DataFrame | pd.Series]:
    calls = load_calls(calls_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top = top_callers(calls)
    tables = {
        "overview": overview(calls),
        "type_semantics": type_semantics(calls),
        "mirrored_records": mirrored_records(calls),
        "duration_summary": duration_summary(calls),
        "duration_by_category": duration_by_category(calls),
        "operator_matrix": operator_matrix(calls),
        "unknown_codes": unknown_codes(calls),
        "activity_distribution": activity_distribution(calls),
        "top_callers": top,
        "top_callers_by_day": daily_stability(calls, top.index),
        "sim_check": sim_check(calls),
        "calls_by_day_hour": calls_by_day_hour(calls),
    }
    with pd.option_context("display.width", 160, "display.max_columns", 30,
                           "display.float_format", "{:,.4g}".format):
        for name, table in tables.items():
            table.to_csv(out_dir / f"{name}.csv")
            if name != "calls_by_day_hour":
                print(f"\n=== {name} ===\n{table.to_string()}")
    for path in save_figures(calls, out_dir):
        print(f"График: {path}")
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EDA очищенных CDR")
    ap.add_argument("--calls", default="data/processed/calls.parquet")
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args()
    run(args.calls, args.out)
