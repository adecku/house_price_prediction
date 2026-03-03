import json
import math
import argparse
from datetime import datetime
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TARGET_COLUMN
from load_data import load_data
from pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ln", "rf"], default="ln")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model == "ln":
        from sklearn.linear_model import LinearRegression
        model = build_pipeline(LinearRegression())
        path = "linear_regression"
    elif args.model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = build_pipeline(RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE, oob_score=True, max_depth=5))
        path = "random_forest"
    else:
        raise ValueError(f"Invalid model: {args.model}")

    data = load_data()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results = {
        "model": args.model,
        "metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        },
    }

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / path
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Baseline LinearRegression metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
