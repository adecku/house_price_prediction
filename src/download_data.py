from pathlib import Path

from sklearn.datasets import fetch_california_housing


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "california_housing.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)
    dataset.frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
