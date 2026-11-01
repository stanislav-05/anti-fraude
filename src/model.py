from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features import FEATURES

SEED = 42
THRESHOLD = 0.5
NEG_MAX_RISK = 20
NEG_MIN_CALLS = 10
IF_MIN_CALLS = 20
SPLIT_DAY = "2026-10-04"

CAT_FEATURES = ["operator"]
DROP = ["is_city", "unknown_code"]
RULE_FEATURES = ["peak_1h", "peak_5m", "uniq_b", "repeat_share", "share_short", "callback_share",
                 "dur_median", "active_days", "share_night", "share_weekend", "n_imsi",
                 "numbers_per_imsi_max"]


def feature_sets(columns) -> dict[str, list[str]]:
    full = [c for c in columns if c not in DROP]
    return {"full": full, "independent": [c for c in full if c not in RULE_FEATURES]}


def make_labels(scores: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """1 — «высокий» по правилам, 0 — уверенно чистый активный номер, NaN — не используем."""

    level = scores["level"].astype(str).reindex(features.index)
    risk = scores["risk"].reindex(features.index).fillna(0)
    y = pd.Series(np.nan, index=features.index)
    y[level.eq("высокий")] = 1
    y[(risk < NEG_MAX_RISK) & features["n_out"].ge(NEG_MIN_CALLS) & y.isna()] = 0
    return y


def prepare_x(features: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = features[cols].copy()
    for c in CAT_FEATURES:
        if c in x:
            x[c] = x[c].astype(str)
    return x


def catboost_model() -> CatBoostClassifier:
    return CatBoostClassifier(iterations=400, depth=6, learning_rate=0.08,
                              auto_class_weights="SqrtBalanced", random_seed=SEED,
                              verbose=False, allow_writing_files=False, thread_count=-1)


def fit(x: pd.DataFrame, y: pd.Series) -> CatBoostClassifier:
    cats = [c for c in CAT_FEATURES if c in x]
    return catboost_model().fit(Pool(x, y.astype(int), cat_features=cats))


def predict(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    cats = [c for c in CAT_FEATURES if c in x]
    return model.predict_proba(Pool(x, cat_features=cats))[:, 1]


def metrics(y: pd.Series, p: np.ndarray, threshold: float = THRESHOLD) -> dict:
    y = y.astype(int).to_numpy()
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {
        "positives": int(y.sum()), "negatives": int((y == 0).sum()),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if y.any() else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "fpr": round(fp / (fp + tn), 7) if fp + tn else None,
    }


def cross_validate(x: pd.DataFrame, y: pd.Series, folds: int = 5) -> np.ndarray:
    """Out-of-fold вероятности: каждый номер оценён моделью, которая его не видела."""

    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for train, test in skf.split(x, y):
        oof[test] = predict(fit(x.iloc[train], y.iloc[train]), x.iloc[test])
    return oof


def temporal_check(calls: pd.DataFrame, split_day: str = SPLIT_DAY) -> dict:
    """Учимся на звонках до split_day, проверяем на звонках после.

    Признаки и метки правил считаются заново для каждого периода, поэтому в тесте
    модель видит номера и поведение, которых не было при обучении.
    """

    from src.features import build_features
    from src.scoring import score

    cut = pd.Timestamp(split_day)
    out = {}
    periods = {}
    for name, part in [("train", calls[calls["ts"] < cut]), ("test", calls[calls["ts"] >= cut])]:
        feats = build_features(part)
        periods[name] = (feats, make_labels(score(feats), feats))
    (f_tr, y_tr), (f_te, y_te) = periods["train"], periods["test"]
    seen = set(f_tr.index[f_tr["n_out"] > 0])
    for name, cols in feature_sets(f_tr.columns).items():
        m_tr, m_te = y_tr.notna(), y_te.notna()
        model = fit(prepare_x(f_tr[m_tr], cols), y_tr[m_tr])
        p = predict(model, prepare_x(f_te[m_te], cols))
        res = metrics(y_te[m_te], p)
        new = ~f_te[m_te].index.isin(list(seen))   # номера, которых не было в обучении
        res["recall_new_numbers"] = metrics(y_te[m_te][new], p[new])["recall"]
        res["positives_new_numbers"] = int(y_te[m_te][new].sum())
        out[name] = res
    out["split_day"] = split_day
    return out


def shap_factors(model: CatBoostClassifier, x: pd.DataFrame, top: int = 4) -> pd.Series:
    """Для каждого номера: «признак = значение (+вклад)», top факторов по модулю вклада.

    SHAP-значения CatBoost аддитивны в логит-шкале: базовое значение + сумма вкладов = логит прогноза.
    """

    cats = [c for c in CAT_FEATURES if c in x]
    shap = model.get_feature_importance(Pool(x, cat_features=cats), type="ShapValues")[:, :-1]
    names = np.array(x.columns)
    order = np.argsort(-np.abs(shap), axis=1)[:, :top]
    out = []
    for i in range(len(x)):
        parts = []
        for j in order[i]:
            v = x.iat[i, j]
            v = f"{v:.3g}" if isinstance(v, (int, float, np.number)) else str(v)
            parts.append(f"{names[j]}={v} ({shap[i, j]:+.2f})")
        out.append("; ".join(parts))
    return pd.Series(out, index=x.index)


def isolation_forest(features: pd.DataFrame, min_calls: int = IF_MIN_CALLS) -> pd.DataFrame:
    """Аномальность номеров с ≥ min_calls исходящими. Объяснение — самые нетипичные признаки.

    Счётчики берём в логарифме (log1p), иначе пара гигантских номеров «сплющит» остальных.
    Объяснение: робастный z-score признака (отклонение от медианы в единицах IQR).
    """

    active = features[features["n_out"] >= min_calls]
    cols = [c for c in feature_sets(features.columns)["full"] if c not in CAT_FEATURES]
    x = active[cols].astype(float)
    counts = [c for c in cols if x[c].max() > 1]
    x[counts] = np.log1p(x[counts].clip(lower=0))
    x = x.fillna(x.median())
    forest = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(x)
    anomaly = -forest.score_samples(x)
    flag = forest.predict(x) == -1
    med, iqr = x.median(), (x.quantile(0.75) - x.quantile(0.25)).replace(0, 1)
    z = ((x - med) / iqr).abs()
    top = z.apply(lambda r: "; ".join(f"{c} ({FEATURES.get(c, c)})" for c in r.nlargest(3).index), axis=1)
    return pd.DataFrame({"if_score": anomaly, "if_anomaly": flag, "if_factors": top}, index=active.index)


def run(processed_dir: str | Path = "data/processed", reports_dir: str | Path = "reports/ml") -> dict:
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(processed_dir / "features.parquet")
    scores = pd.read_parquet(processed_dir / "scores.parquet")
    y = make_labels(scores, features)
    labeled = y.notna()
    sets = feature_sets(features.columns)
    report: dict = {"labels": {"positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
                               "unlabeled_callers": int((~labeled & features["n_out"].gt(0)).sum())}}
    
    calls_path = processed_dir / "calls.parquet"
    if calls_path.exists():
        print("Проверка по дням: обучение 1–3 октября, тест 4–5 октября…")
        calls = pd.read_parquet(calls_path, columns=["ctype", "imsi", "a", "b", "ts", "dur", "c"])
        report["temporal"] = temporal_check(calls)
    else:
        print("calls.parquet нет — проверка по дням пропущена, только кросс-валидация")

    print("5-кратная кросс-валидация…")
    oof = {}
    report["cv"] = {}
    for name, cols in sets.items():
        oof[name] = cross_validate(prepare_x(features[labeled], cols), y[labeled])
        report["cv"][name] = metrics(y[labeled], oof[name])

    callers = features[features["n_out"] > 0]
    model = fit(prepare_x(features[labeled], sets["full"]), y[labeled])
    model.save_model(str(processed_dir / "catboost_full.cbm"))
    x_all = prepare_x(callers, sets["full"])
    ml = pd.DataFrame({"ml_proba": predict(model, x_all)}, index=callers.index)
    ml["ml_proba_oof"] = pd.Series(oof["full"], index=y[labeled].index)  # выравнивание по номеру
    ml["ml_proba_independent_oof"] = pd.Series(oof["independent"], index=y[labeled].index)
    interesting = ml.index[(ml["ml_proba"] >= 0.1) | scores["level"].reindex(ml.index).astype(str).ne("низкий")]
    ml.loc[interesting, "shap_factors"] = shap_factors(model, x_all.loc[interesting])
    print("Isolation Forest…")
    iso = isolation_forest(features)
    ml = ml.join(iso, how="left")
    ml = ml.join(scores[["risk", "level", "pattern"]], how="left")
    ml["rules_high"] = ml["level"].astype(str).eq("высокий")
    ml["ml_high"] = ml["ml_proba"] >= THRESHOLD
    ml.drop(columns=["risk", "level", "pattern"]).to_parquet(processed_dir / "ml_scores.parquet")
    lab = ml.loc[y[labeled].index.intersection(ml.index)]
    oof_high = lab["ml_proba_oof"] >= THRESHOLD
    report["agreement_oof"] = {
        "both": int((lab["rules_high"] & oof_high).sum()),
        "rules_only": int((lab["rules_high"] & ~oof_high).sum()),
        "model_only": int((~lab["rules_high"] & oof_high).sum()),
    }
    unl = ml[~ml.index.isin(y[labeled].index)]
    report["unlabeled_model_high"] = int((unl["ml_proba"] >= THRESHOLD).sum())
    legit = scores.index[scores["legit_signs"].fillna(False) & scores["n_out"].ge(500)]
    report["call_centers"] = {"count": len(legit),
                              "flagged_by_model_oof": int((ml.loc[ml.index.intersection(legit), "ml_proba_oof"] >= THRESHOLD).sum()),
                              "if_anomalies": int(ml.loc[ml.index.intersection(legit), "if_anomaly"].fillna(False).sum())}

    an = ml[ml["if_anomaly"].fillna(False)]
    report["isolation_forest"] = {
        "active_numbers": len(iso), "anomalies": len(an),
        "anomalies_rules_high": int(an["rules_high"].sum()),
        "rules_high_caught": int(an["rules_high"].sum()), "rules_high_total": int(ml["rules_high"].sum()),
        "anomalies_not_in_rules": int((~an["rules_high"] & an["level"].astype(str).eq("низкий")).sum()),
    }
    cols = ["risk", "level", "pattern", "ml_proba", "ml_proba_oof", "ml_proba_independent_oof", "shap_factors"]
    dis = ml[(ml["rules_high"] != (ml["ml_proba_oof"].fillna(ml["ml_proba"]) >= THRESHOLD))
             | (ml["level"].astype(str).eq("проверка"))]
    dis[cols].sort_values("ml_proba", ascending=False).to_csv(reports_dir / "disagreements.csv", encoding="utf-8-sig")
    new = an[~an["rules_high"] & an["level"].astype(str).eq("низкий")].sort_values("if_score", ascending=False)
    new = new.join(features[["n_out", "uniq_b", "dur_median", "share_short", "peak_1h", "active_days"]])
    new.to_csv(reports_dir / "anomalies.csv", encoding="utf-8-sig")
    (reports_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report, dis[cols], new, model, sets["full"])
    return report


def print_report(report: dict, dis: pd.DataFrame, new: pd.DataFrame, model, cols) -> None:
    def table(d: dict) -> str:
        return pd.DataFrame(d).T[["positives", "negatives", "pr_auc", "precision", "recall", "fp", "fpr"]
                                 + [c for c in ["recall_new_numbers", "positives_new_numbers"]
                                    if c in next(iter(d.values()))]].to_string()

    print(f"\nМетки: {report['labels']}")
    if "temporal" in report:
        t = {k: v for k, v in report["temporal"].items() if isinstance(v, dict)}
        print(f"\n=== Проверка по дням (тест с {report['temporal']['split_day']}) ===\n{table(t)}")
    print(f"\n=== Кросс-валидация, порог {THRESHOLD} ===\n{table(report['cv'])}")
    print(f"\nСогласие правил и модели (out-of-fold): {report['agreement_oof']}")
    print(f"Номера вне обучения, которые модель считает фродом: {report['unlabeled_model_high']}")
    print(f"Колл-центры: {report['call_centers']}")
    print(f"Isolation Forest: {report['isolation_forest']}")
    imp = pd.Series(model.get_feature_importance(), index=cols).sort_values(ascending=False).head(10)
    print(f"\nВажность признаков (full):\n{imp.round(1).to_string()}")
    with pd.option_context("display.width", 220, "display.max_colwidth", 120):
        print(f"\nРасхождения правил и модели ({len(dis)}):")
        print(dis.head(15).to_string())
        print(f"\nАномалии IF, которых нет в правилах ({len(new)}), первые 10:")
        print(new.head(10)[["if_score", "n_out", "uniq_b", "dur_median", "share_short", "if_factors"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CatBoost + Isolation Forest")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="reports/ml")
    args = ap.parse_args()
    run(args.processed, args.out)
