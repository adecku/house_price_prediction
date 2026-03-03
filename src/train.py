import json
import math
import argparse
from datetime import datetime
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split

from config import RANDOM_STATE, TARGET_COLUMN
from load_data import load_data
from model_tuning import tune_random_forest
from pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lr", "rfr"], default="lr")
    parser.add_argument("--tune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    best_params = None

    if args.model == "lr":
        from sklearn.linear_model import LinearRegression
        model = build_pipeline(LinearRegression())
        path = "linear_regression"
    elif args.model == "rfr":
        from sklearn.ensemble import RandomForestRegressor
        model = build_pipeline(RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE))
        path = "random_forest"
    else:
        raise ValueError(f"Invalid model: {args.model}")

    data = load_data()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    if args.tune:
        if args.model != "rfr":
            raise ValueError("--tune is available only for model='rfr'")
        model, best_params = tune_random_forest(X_train, y_train)

    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=("neg_mean_absolute_error", "neg_mean_squared_error", "r2"),
        return_train_score=False,
    )

    cv_mae_scores = [-value for value in cv_scores["test_neg_mean_absolute_error"]]
    cv_mse_scores = [-value for value in cv_scores["test_neg_mean_squared_error"]]
    cv_rmse_scores = [math.sqrt(value) for value in cv_mse_scores]
    cv_r2_scores = list(cv_scores["test_r2"])

    cv_metrics = {
        "mae_mean": sum(cv_mae_scores) / len(cv_mae_scores),
        "mae_std": float(cv_scores["test_neg_mean_absolute_error"].std()),
        "mse_mean": sum(cv_mse_scores) / len(cv_mse_scores),
        "mse_std": float(cv_scores["test_neg_mean_squared_error"].std()),
        "rmse_mean": sum(cv_rmse_scores) / len(cv_rmse_scores),
        "rmse_std": (sum((value - (sum(cv_rmse_scores) / len(cv_rmse_scores))) ** 2 for value in cv_rmse_scores) / len(cv_rmse_scores)) ** 0.5,
        "r2_mean": sum(cv_r2_scores) / len(cv_r2_scores),
        "r2_std": float(cv_scores["test_r2"].std()),
        "cv_folds": 5,
    }

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results = {
        "model": args.model,
        "tuned": args.tune,
        "metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        },
        "cv_metrics": cv_metrics,
    }
    if best_params is not None:
        results["best_params"] = best_params

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
    print("Cross-validation metrics (5-fold):")
    print(f"MAE: {cv_metrics['mae_mean']:.4f} +/- {cv_metrics['mae_std']:.4f}")
    print(f"MSE: {cv_metrics['mse_mean']:.4f} +/- {cv_metrics['mse_std']:.4f}")
    print(f"RMSE: {cv_metrics['rmse_mean']:.4f} +/- {cv_metrics['rmse_std']:.4f}")
    print(f"R2: {cv_metrics['r2_mean']:.4f} +/- {cv_metrics['r2_std']:.4f}")
    if best_params is not None:
        print(f"Best params: {best_params}")
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
