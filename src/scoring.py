from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.policy import EMERGENCY_NUMBERS, is_whitelisted, load_whitelist

MIN_CALLS = 100
HIGH = 70
REVIEW = 40


@dataclass(frozen=True)
class Rule:
    name: str
    points: int
    text: str
    condition: Callable[[pd.DataFrame], pd.Series]


RULES = [
    Rule("burst_1h", 40, "{peak_1h:.0f} звонков за час",
         lambda f: f["peak_1h"] >= 300),
    Rule("burst_5m", 15, "{peak_5m:.0f} звонков за 5 минут",
         lambda f: f["peak_5m"] >= 50),
    Rule("mass_unique", 15, "{uniq_b:.0f} разных получателей, повторов {repeat_share:.0%}",
         lambda f: (f["uniq_b"] >= 300) & (f["repeat_share"] <= 0.05)),
    Rule("short_calls", 10, "коротких звонков (до 10 с): {share_short:.0%}",
         lambda f: (f["n_out"] >= MIN_CALLS) & (f["share_short"] >= 0.3)),
    Rule("no_callbacks", 10, "никто из {uniq_b:.0f} получателей не перезвонил",
         lambda f: (f["n_out"] >= MIN_CALLS) & (f["callback_share"] == 0)),
    Rule("robot_profile", 35,
         "профиль робота: медиана {dur_median:.0f} с, все получатели разные, перезвонов нет",
         lambda f: (f["uniq_b"] >= 300) & (f["repeat_share"] <= 0.05)
         & (f["callback_share"] == 0) & (f["dur_median"] <= 20)),
    Rule("wangiri", 45, "{share_short:.0%} звонков короче 10 с — «звонок-сброс» (wangiri)",
         lambda f: (f["uniq_b"] >= 300) & (f["share_short"] >= 0.9)),
    Rule("round_clock", 35, "звонит круглосуточно: {active_days:.0f} дн., ночью {share_night:.0%}",
         lambda f: (f["n_out"] >= 500) & (f["active_days"] >= 4) & (f["share_night"] >= 0.1)),
    Rule("many_sims", 70, "номер работал с {n_imsi:.0f} разными SIM-картами",
         lambda f: f["n_imsi"] >= 3),
    Rule("shared_sim", 10, "одну SIM делят {numbers_per_imsi_max:.0f} номера",
         lambda f: f["numbers_per_imsi_max"] >= 2),
    Rule("office_schedule", -45,
         "график офиса: {active_days:.0f} будних дня, ночью и в выходные звонков нет",
         lambda f: (f["active_days"] >= 3) & (f["share_weekend"] == 0) & (f["share_night"] == 0)),
    Rule("callbacks", -20, "{callback_share:.0%} получателей перезванивают",
         lambda f: (f["n_out"] >= MIN_CALLS) & (f["callback_share"] >= 0.1)
         & (f["share_short"] < 0.9)),
]


def fired_rules(features: pd.DataFrame) -> pd.DataFrame:
    """Таблица True/False: какое правило сработало у какого номера."""

    return pd.DataFrame(
        {r.name: r.condition(features).fillna(False).astype(bool) for r in RULES},
        index=features.index,
    )


def pattern_of(fired: pd.DataFrame) -> pd.Series:
    """Короткая метка паттерна для ответа и отчёта (первое подходящее по приоритету)."""

    pattern = pd.Series("", index=fired.index, dtype="str")
    for name, label in [("many_sims", "multi_sim"), ("round_clock", "round_clock"),
                        ("burst_1h", "burst"), ("wangiri", "wangiri")]:
        pattern = pattern.mask(fired[name], label)
    return pattern


def explain(features: pd.DataFrame, fired: pd.DataFrame) -> pd.Series:
    """Объяснение: «+40 2231 звонков за час; +15 …» по убыванию вклада.

    Строки собираем циклом, но только для номеров с сработавшими правилами — их мало.
    """

    rules = sorted(RULES, key=lambda r: -abs(r.points))
    rows = fired.index[fired.any(axis=1)]
    out = {}
    for number in rows:
        f = features.loc[number]
        parts = [f"{r.points:+d} {r.text.format(**f)}" for r in rules if fired.at[number, r.name]]
        out[number] = "; ".join(parts)
    return pd.Series(out, dtype="str").reindex(features.index, fill_value="")


def score(features: pd.DataFrame, whitelist: frozenset[str] = EMERGENCY_NUMBERS) -> pd.DataFrame:
    fired = fired_rules(features)
    white = pd.Series(is_whitelisted(features.index, whitelist), index=features.index)
    fired = fired.mask(white, False)
    points = pd.Series({r.name: r.points for r in RULES})
    risk = (fired.astype(int) @ points).clip(0, 100)
    level = pd.cut(risk, bins=[-1, REVIEW - 1, HIGH - 1, 100],
                   labels=["низкий", "проверка", "высокий"])
    return pd.DataFrame({
        "risk": risk,
        "level": level,
        "pattern": pattern_of(fired),
        "legit_signs": fired[[r.name for r in RULES if r.points < 0]].any(axis=1),
        "reasons": explain(features, fired).mask(white, "белый список: экстренная/доверенная служба"),
        "whitelisted": white,
        "n_out": features["n_out"],
        "operator": features["operator"],
    }).sort_values("risk", ascending=False)


def write_answer(scores: pd.DataFrame, path: str | Path, include_review: bool = False) -> pd.Series:
    """answer.txt: номер;паттерн, по одному на строку. Проверяет формат перед записью."""

    levels = ["высокий", "проверка"] if include_review else ["высокий"]
    chosen = scores[scores["level"].isin(levels)]
    numbers = chosen.index.to_series()
    assert numbers.str.fullmatch(r"\d{10}").all(), "в ответе номер не из 10 цифр"
    assert numbers.is_unique, "в ответе повторяются номера"
    if "whitelisted" in chosen:
        assert not chosen["whitelisted"].any(), "в ответ попал номер из белого списка"
    lines = [
        "# Антифрод по метаданным CDR, 1–5 октября 2026",
        f"# номеров: {len(chosen)}; уровни: {', '.join(levels)}",
        *(numbers + ";" + chosen["pattern"].replace("", "mass")),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return numbers


def run(processed_dir: str | Path = "data/processed", answer_path: str | Path = "answer.txt",
        reports_dir: str | Path = "reports", include_review: bool = False) -> pd.DataFrame:
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(processed_dir / "features.parquet")
    scores = score(features, load_whitelist())
    scores.to_parquet(processed_dir / "scores.parquet")
    candidates = scores[scores["reasons"] != ""]
    candidates.to_csv(reports_dir / "candidates.csv", encoding="utf-8-sig")
    answer = write_answer(scores, answer_path, include_review)
    print("Номеров по уровням риска:")
    print(scores["level"].value_counts().sort_index(ascending=False).to_string())
    print("\nПаттерны среди уровней «высокий» и «проверка»:")
    top = scores[scores["level"].isin(["высокий", "проверка"])]
    print(pd.crosstab(top["pattern"].replace("", "mass"), top["level"].astype(str)).to_string())
    with pd.option_context("display.width", 200, "display.max_colwidth", 140):
        print("\nПримеры «высокий»:")
        print(top[top["level"] == "высокий"].head(5)[["risk", "pattern", "reasons"]].to_string())
        print("\nВсе «проверка»:")
        print(top[top["level"] == "проверка"][["risk", "pattern", "reasons"]].to_string())
        legit = scores[scores["legit_signs"] & (scores["n_out"] >= 500)]
        print(f"\nКрупные номера с признаками легального источника (не в ответе): {len(legit)}")
        print(legit.head(10)[["risk", "n_out", "operator", "reasons"]].to_string())

    print(f"\nОтвет: {answer_path} — {len(answer)} номеров; кандидаты: {reports_dir / 'candidates.csv'}")
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Скоринг номеров и файл ответа")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--answer", default="answer.txt")
    ap.add_argument("--include-review", action="store_true",
                    help="включить в ответ уровень «проверка»")
    args = ap.parse_args()
    run(args.processed, args.answer, include_review=args.include_review)
