from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

CANDIDATE_LEVELS = ["высокий", "проверка"]
STRONG_OVERLAP = 0.3


def candidates(scores: pd.DataFrame) -> pd.DataFrame:
    """Номера уровней «высокий» и «проверка» с паттерном (пустой паттерн -> mass)."""

    c = scores[scores["level"].astype(str).isin(CANDIDATE_LEVELS)].copy()
    c["pattern"] = c["pattern"].replace("", "mass")
    return c


def sim_links(calls: pd.DataFrame) -> pd.DataFrame:
    """Пары (номер, IMSI), где SIM общая у нескольких номеров или у номера несколько SIM.

    Обычные пары «один номер — одна SIM» связей не создают, поэтому их отбрасываем:
    граф остаётся маленьким.
    """
    
    pairs = calls.loc[calls["imsi"].notna(), ["a", "imsi"]].drop_duplicates()
    numbers_per_sim = pairs.groupby("imsi")["a"].transform("nunique")
    sims_per_number = pairs.groupby("a")["imsi"].transform("nunique")
    return pairs[(numbers_per_sim > 1) | (sims_per_number > 1)].reset_index(drop=True)


def recipient_overlap(calls: pd.DataFrame, numbers: pd.Index) -> pd.DataFrame:
    """Все пары кандидатов, у которых есть хотя бы один общий получатель.

    Приём: self-merge по b. Каждая строка результата — «получатель, которому
    звонили оба номера». Потом считаем такие строки для каждой пары.
    """

    pairs = calls.loc[calls["a"].isin(numbers), ["a", "b"]].drop_duplicates()
    size = pairs.groupby("a").size()
    both = pairs.merge(pairs, on="b", suffixes=("_1", "_2"))
    both = both[both["a_1"] < both["a_2"]]
    shared = both.groupby(["a_1", "a_2"]).size().rename("shared").reset_index()
    n1, n2 = shared["a_1"].map(size), shared["a_2"].map(size)
    shared["jaccard"] = shared["shared"] / (n1 + n2 - shared["shared"])
    return shared.sort_values("jaccard", ascending=False, ignore_index=True)


def pattern_overlap(calls: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Для каждого паттерна: насколько аудитории его номеров пересекаются.

    overlap_share = 1 - (уникальных получателей) / (сумма получателей по номерам).
    0 - каждый номер обзванивал свой, ни с кем не общий список (признак веера).
    """

    pairs = calls.loc[calls["a"].isin(cand.index), ["a", "b"]].drop_duplicates()
    pairs = pairs.assign(pattern=pairs["a"].map(cand["pattern"]))
    g = pairs.groupby("pattern")
    out = pd.DataFrame({
        "numbers": g["a"].nunique(),
        "recipients_sum": g.size(),
        "recipients_unique": g["b"].nunique(),
    })
    out["overlap_share"] = 1 - out["recipients_unique"] / out["recipients_sum"]
    out["recipients_per_number"] = out["recipients_sum"] / out["numbers"]
    return out.sort_values("numbers", ascending=False)


def victims_hit_twice(calls: pd.DataFrame, numbers: pd.Index) -> pd.Series:
    """Получатели, до которых дозванивались 2+ разных подозрительных номера."""

    pairs = calls.loc[calls["a"].isin(numbers), ["a", "b"]].drop_duplicates()
    per_victim = pairs.groupby("b")["a"].nunique()
    return per_victim[per_victim > 1].sort_values(ascending=False)


def schedule(calls: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Паттерн × день: сколько подозрительных номеров звонили в этот день."""

    c = calls.loc[calls["a"].isin(cand.index), ["a", "ts"]]
    c = c.assign(day=c["ts"].dt.strftime("%d.%m"), pattern=c["a"].map(cand["pattern"]))
    return c.pivot_table(index="pattern", columns="day", values="a",
                         aggfunc="nunique", fill_value=0)


def build_groups(links: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    """Граф: номер — SIM (из links) + номер — номер (сильное пересечение получателей).

    Компонента связности с 2+ номерами = группа. Возвращает number -> group_id, group_size.
    """

    g = nx.Graph()
    g.add_edges_from(zip("n:" + links["a"], "s:" + links["imsi"]))
    strong = overlap[overlap["jaccard"] >= STRONG_OVERLAP]
    g.add_edges_from(zip("n:" + strong["a_1"], "n:" + strong["a_2"]))
    rows = []
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    for group_id, nodes in enumerate(components, start=1):
        numbers = sorted(n[2:] for n in nodes if n.startswith("n:"))
        if len(numbers) < 2:
            continue  # номер с несколькими SIM, ни с кем не связанный, - не группа
        rows += [(n, group_id, len(numbers)) for n in numbers]
    return pd.DataFrame(rows, columns=["number", "group_id", "group_size"]).set_index("number")


def describe_groups(groups: pd.DataFrame, calls: pd.DataFrame, links: pd.DataFrame,
                    scores: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на группу: состав, паттерны, охват и время активности."""

    if groups.empty:
        return pd.DataFrame()
    c = calls.loc[calls["a"].isin(groups.index), ["a", "b", "ts"]]
    c = c.assign(group_id=c["a"].map(groups["group_id"]))
    sims = links.assign(group_id=links["a"].map(groups["group_id"])).dropna(subset=["group_id"])
    info = groups.join(scores[["level", "pattern"]], how="left")
    info["level"] = info["level"].astype(str).replace("nan", "низкий")
    info["label"] = info["pattern"].fillna("").replace("", "mass").where(
        info["level"].isin(CANDIDATE_LEVELS), "низкий риск")

    g = c.groupby("group_id")
    out = pd.DataFrame({
        "numbers": info.groupby("group_id").apply(lambda d: ", ".join(d.index)),
        "size": groups.groupby("group_id").size(),
        "candidates": info.groupby("group_id")["level"].apply(lambda s: s.isin(CANDIDATE_LEVELS).sum()),
        "composition": info.groupby("group_id")["label"].apply(
            lambda s: ", ".join(f"{k}×{v}" for k, v in s.value_counts().items())),
        "shared_sims": sims.groupby("group_id")["imsi"].nunique(),
        "calls": g.size(),
        "recipients": g["b"].nunique(),
        "first": g["ts"].min(),
        "last": g["ts"].max(),
    })
    out[["shared_sims", "calls", "recipients"]] = out[["shared_sims", "calls", "recipients"]].fillna(0).astype(int)
    return out.sort_values(["candidates", "size"], ascending=False)


def run(processed_dir: str | Path = "data/processed", reports_dir: str | Path = "reports/graph") -> dict:
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    calls = pd.read_parquet(processed_dir / "calls.parquet", columns=["a", "b", "imsi", "ts"])
    scores = pd.read_parquet(processed_dir / "scores.parquet")
    cand = candidates(scores)
    links = sim_links(calls)
    overlap = recipient_overlap(calls, cand.index)
    groups = build_groups(links, overlap)
    result = {
        "pattern_overlap": pattern_overlap(calls, cand),
        "schedule": schedule(calls, cand),
        "recipient_overlap": overlap,
        "groups": describe_groups(groups, calls, links, scores),
    }
    groups.to_parquet(processed_dir / "groups.parquet")
    for name, table in result.items():
        table.to_csv(reports_dir / f"{name}.csv", encoding="utf-8-sig")

    twice = victims_hit_twice(calls, cand.index)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80,
                           "display.float_format", "{:,.3f}".format):
        print(f"Кандидатов: {len(cand)}; связей номер—SIM: {len(links)}")
        print("\n=== Пересечение аудиторий внутри паттернов (0 = списки не пересекаются) ===")
        print(result["pattern_overlap"].to_string())
        print(f"\nЖертв, которым звонили 2+ подозрительных номера: {len(twice):,}")
        print(f"Пар кандидатов с общими получателями: {len(overlap):,}; "
              f"из них сильно пересекаются (Жаккар >= {STRONG_OVERLAP}): "
              f"{(overlap['jaccard'] >= STRONG_OVERLAP).sum():,}")
        if len(overlap):
            print(overlap.head(10).to_string(index=False))
        print("\n=== Расписание: сколько номеров каждого паттерна работало в день ===")
        print(result["schedule"].to_string())
        gr = result["groups"]
        print(f"\n=== Группы связанных номеров: {len(gr)}, "
              f"из них с кандидатами: {(gr['candidates'] > 0).sum() if len(gr) else 0} ===")
        if len(gr):
            print(gr.drop(columns="numbers").head(20).to_string())
    print(f"\nСохранено: {processed_dir / 'groups.parquet'}, {reports_dir}/*.csv")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Графовый анализ кандидатов")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="reports/graph")
    args = ap.parse_args()
    run(args.processed, args.out)
