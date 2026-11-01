from src import call_risk, eda, features, graph, model, preprocess, scoring


def main() -> None:
    print("1/7 Очистка CDR")
    preprocess.run("data/raw", "data/processed")
    print("\n2/7 Признаки по номерам")
    features.run("data/processed")
    print("\n3/7 EDA")
    eda.run("data/processed/calls.parquet", "reports/eda")
    print("\n4/7 Скоринг и ответ")
    scoring.run("data/processed", "answer.txt", "reports")
    print("\n5/7 Графовый анализ")
    graph.run("data/processed", "reports/graph")
    print("\n6/7 Риск отдельных звонков")
    call_risk.run("data/processed", "reports")
    print("\n7/7 ML: CatBoost и Isolation Forest")
    model.run("data/processed", "reports/ml")


if __name__ == "__main__":
    main()
