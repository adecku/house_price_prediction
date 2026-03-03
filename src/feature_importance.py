import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from config import RANDOM_STATE, TARGET_COLUMN
from load_data import load_data
from pipeline import build_pipeline


def main() -> None:
    data = load_data()
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    model = build_pipeline(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE))
    model.fit(X, y)

    preprocessor = model.named_steps["preprocessing"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    feature_importance = [
        {"feature": name, "importance": float(value)}
        for name, value in zip(feature_names, importances)
    ]
    feature_importance.sort(key=lambda item: item["importance"], reverse=True)

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "feature_importance.json"
    png_path = output_dir / "feature_importance.png"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(feature_importance, file, indent=2)

    labels = [item["feature"] for item in feature_importance]
    values = [item["importance"] for item in feature_importance]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Saved JSON: {json_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
