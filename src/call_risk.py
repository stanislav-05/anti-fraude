from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.policy import AUTOBLOCK, EMERGENCY_NUMBERS, is_whitelisted, load_whitelist, recommendation
from src.scoring import HIGH, REVIEW

CALL_TYPE = {"01": "входящий", "02": "исходящий", "03": "переадресация"}
MONTHS = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


@dataclass(frozen=True)
class CallRule:
    name: str
    points: int
    text: str
    condition: Callable[[pd.DataFrame], pd.Series]


CALL_RULES = [
    CallRule("callback_to_wangiri", 70,
             "перезвон на номер схемы wangiri — вероятно, платный номер",
             lambda m: m["b_pattern"].eq("wangiri") & m["b_level"].eq("высокий")),
    CallRule("call_to_suspect", 30, "звонок на подозрительный номер ({b_pattern})",
             lambda m: m["b_level"].eq("высокий") & m["b_pattern"].ne("wangiri")),
    CallRule("drop_2s", 10, "сброс на первых секундах: {dur} с",
             lambda m: m["dur"].le(2).fillna(False)),
    CallRule("short", 5, "звонок короче 10 с: {dur} с",
             lambda m: m["dur"].between(3, 10).fillna(False)),
    CallRule("night", 5, "ночной звонок: {hour} ч",
             lambda m: m["hour"].lt(6)),
    CallRule("no_contact", 5, "получатель ни разу не звонил этому номеру",
             lambda m: ~m["known_contact"]),
    CallRule("group", 5, "номер из связанной группы: {group_size} номеров",
             lambda m: m["group_size"].ge(2) & m["a_risk"].ge(REVIEW)),
]


def enrich(calls: pd.DataFrame, scores: pd.DataFrame, groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Добавляет к звонкам всё, что нужно правилам: риск A, паттерн B, контакт, группа."""

    m = calls[["ctype", "a", "b", "ts", "dur"]].copy()
    m["ctype"] = m["ctype"].astype(str)
    m["hour"] = m["ts"].dt.hour
    m["a_risk"] = m["a"].map(scores["risk"]).fillna(0).astype(int)
    m["a_level"] = m["a"].map(scores["level"].astype(str)).fillna("низкий")
    m["a_pattern"] = m["a"].map(scores["pattern"]).fillna("")
    m["b_level"] = m["b"].map(scores["level"].astype(str)).fillna("низкий")
    m["b_pattern"] = m["b"].map(scores["pattern"].replace("", "mass")).fillna("")
    pairs = pd.MultiIndex.from_frame(m[["a", "b"]].drop_duplicates())
    m["known_contact"] = pd.MultiIndex.from_arrays([m["b"], m["a"]]).isin(pairs)
    size = groups["group_size"] if groups is not None and len(groups) else pd.Series(dtype=int)
    m["group_size"] = m["a"].map(size).fillna(1).astype(int)
    return m


def fired_call_rules(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({r.name: r.condition(m).astype(bool) for r in CALL_RULES}, index=m.index)


def score_calls(calls: pd.DataFrame, scores: pd.DataFrame, groups: pd.DataFrame | None = None,
                whitelist: frozenset[str] = EMERGENCY_NUMBERS) -> pd.DataFrame:
    m = enrich(calls, scores, groups)
    fired = fired_call_rules(m)
    points = pd.Series({r.name: r.points for r in CALL_RULES})
    m["call_points"] = fired.astype(int) @ points
    m["risk"] = (m["a_risk"] + m["call_points"]).clip(0, 100)
    white = is_whitelisted(m["a"], whitelist) | is_whitelisted(m["b"], whitelist)
    m["whitelisted"] = white
    m.loc[white, ["risk", "call_points"]] = 0
    m["level"] = pd.cut(m["risk"], bins=[-1, REVIEW - 1, HIGH - 1, 100],
                        labels=["низкий", "проверка", "высокий"])
    m["recommendation"] = recommendation(m["level"]).astype("category")
    m["autoblock"] = AUTOBLOCK
    return m


def mask_number(n: str) -> str:
    """0252432858 -> +7 025 ***-**-58; короткие служебные номера показываем как есть."""

    return f"+7 {n[:3]} ***-**-{n[-2:]}" if isinstance(n, str) and len(n) == 10 else str(n)


def _bar(points: int, scale: int = 4) -> str:
    return ("█" if points > 0 else "░") * max(1, abs(points) // scale)


def card(scored: pd.DataFrame, idx, scores: pd.DataFrame) -> str:
    """Текстовая карточка звонка scored.loc[idx]: риск, факторы с баллами, рекомендация."""

    row = scored.loc[idx]
    ts = row["ts"]
    dur = "длительность неизвестна" if pd.isna(row["dur"]) else f"{int(row['dur'])} сек"
    lines = [
        f"{mask_number(row['a'])} → {mask_number(row['b'])}",
        f"{ts.day} {MONTHS[ts.month - 1]}, {ts:%H:%M:%S} · {dur} · {CALL_TYPE.get(row['ctype'], row['ctype'])}",
        f"РИСК {int(row['risk'])}",
        "Что дало этот балл:",
    ]
    if row["whitelisted"]:
        lines.append("   0  белый список: экстренная/доверенная служба — звонок не оценивается")
    else:
        factors: list[tuple[int, str]] = []
        reasons = scores["reasons"].get(row["a"], "") if row["a_risk"] > 0 else ""
        for part in filter(None, reasons.split("; ")):
            pts, text = part.split(" ", 1)
            factors.append((int(pts), f"номер A: {text}"))
        fired = fired_call_rules(scored.loc[[idx]]).iloc[0]
        values = {**row.to_dict(), "dur": "?" if pd.isna(row["dur"]) else int(row["dur"])}
        for r in CALL_RULES:
            if fired[r.name]:
                factors.append((r.points, r.text.format(**values)))
        factors.sort(key=lambda f: -abs(f[0]))
        for pts, text in factors:
            lines.append(f"  {pts:+4d}  {text:<70} {_bar(pts)}")
        total = sum(p for p, _ in factors)
        if total != row["risk"]:
            lines.append(f"  сумма факторов {total}, риск ограничен диапазоном 0–100")
    lines.append(f"рекомендация: {row['recommendation']} · автоблокировка: {'да' if row['autoblock'] else 'нет'}")
    return "\n".join(lines)


def pick_examples(scored: pd.DataFrame, scores: pd.DataFrame) -> dict[str, int]:
    """По одному показательному звонку на каждый случай для защиты."""

    ex: dict[str, int] = {}
    for pattern in ["burst", "wangiri", "mass", "multi_sim", "round_clock"]:
        sel = scored[scored["a_pattern"].replace("", "mass").eq(pattern) & scored["a_level"].ne("низкий")]
        if len(sel):
            ex[f"номер схемы {pattern}"] = sel["risk"].idxmax()
    callback = scored[scored["b_pattern"].eq("wangiri") & scored["b_level"].eq("высокий")]
    if len(callback):
        ex["жертва перезванивает на номер wangiri"] = callback.index[0]
    legit = scores.index[scores["legit_signs"] & (scores["n_out"] >= 500)]
    sel = scored[scored["a"].isin(legit)]
    if len(sel):
        ex["колл-центр (легальный массовый источник)"] = sel.index[0]
    sel = scored[scored["a_risk"].eq(0) & scored["dur"].gt(60).fillna(False)]
    if len(sel):
        ex["обычный звонок"] = sel.index[0]
    return ex


def run(processed_dir: str | Path = "data/processed", reports_dir: str | Path = "reports",
        number: str | None = None) -> pd.DataFrame:
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    calls = pd.read_parquet(processed_dir / "calls.parquet", columns=["ctype", "a", "b", "ts", "dur"])
    scores = pd.read_parquet(processed_dir / "scores.parquet")
    groups_path = processed_dir / "groups.parquet"
    groups = pd.read_parquet(groups_path) if groups_path.exists() else None
    scored = score_calls(calls, scores, groups, load_whitelist())
    if number:
        sel = scored[scored["a"].eq(number) | scored["b"].eq(number)].nlargest(3, "risk")
        for idx in sel.index:
            print(card(scored, idx, scores), end="\n\n")
        return scored
    
    scored.drop(columns=["hour", "a_level", "b_level", "known_contact"]).to_parquet(
        processed_dir / "call_scores.parquet", index=False)

    print("Звонков по уровням риска:")
    counts = scored["level"].value_counts().sort_index(ascending=False)
    print(pd.DataFrame({"звонков": counts, "доля": (counts / len(scored)).round(5)}).to_string())
    print(f"Звонков с/на номера из белого списка: {int(scored['whitelisted'].sum())}; "
          f"автоблокировок: {int(scored['autoblock'].sum())}")
    md = ["# Примеры карточек звонков\n"]
    for title, idx in pick_examples(scored, scores).items():
        text = card(scored, idx, scores)
        print(f"\n=== {title} ===\n{text}")
        md += [f"## {title}\n", "```", text, "```\n"]
    (reports_dir / "call_cards.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nСохранено: {processed_dir / 'call_scores.parquet'}, {reports_dir / 'call_cards.md'}")
    return scored


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Риск отдельного вызова и карточки звонков")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--number", help="показать карточки звонков этого номера")
    args = ap.parse_args()
    run(args.processed, number=args.number)
