from __future__ import annotations

from pathlib import Path

import pandas as pd

AUTOBLOCK = False

EMERGENCY_NUMBERS = frozenset({
    "112",
    "101", "102", "103", "104",
    "01", "02", "03", "04",
})
WHITELIST_FILE = Path("data/whitelist.txt")
RECOMMENDATIONS = {
    "высокий": "пометить абоненту: «возможно, мошенники»",
    "проверка": "на ручную проверку аналитику",
    "низкий": "пропустить без действий",
}


def load_whitelist(path: str | Path | None = WHITELIST_FILE) -> frozenset[str]:
    """Экстренные номера + доверенные номера из файла (если он есть)."""

    extra: set[str] = set()
    if path is not None and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                extra.add(line)
    return EMERGENCY_NUMBERS | extra


def is_whitelisted(numbers: pd.Series | pd.Index, whitelist: frozenset[str] = EMERGENCY_NUMBERS):
    """True для номеров из белого списка (работает и для Series, и для Index)."""
    
    return numbers.isin(list(whitelist))


def recommendation(level: pd.Series) -> pd.Series:
    return level.astype(str).map(RECOMMENDATIONS)
