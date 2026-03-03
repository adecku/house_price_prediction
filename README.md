# House Price Prediction

A simple ML project for predicting house prices on the **California Housing** dataset.

## What is included in the project

- data download to CSV (`src/download_data.py`)
- data loading (`src/load_data.py`)
- simple feature engineering (`src/features.py`)
- pipeline: feature engineering + preprocessing + model (`src/pipeline.py`)
- training and saving metrics to JSON (`src/train.py`)

## Structure

- `data/raw/` - input CSV data
- `notebooks/` - folder with Jupyter notebooks
- `src/` - source code
- `outputs/` - training results (metrics in JSON)

## Installation

```bash
pip install -r requirements.txt
```

## Run

1. Download data:

```bash
python src/download_data.py
```

2. Training:

```bash
python src/train.py --model {model_name}
```

Available models:
- Linear Regression (`lr`)
- Random Forest Regression (`rfr`)

## Results

After training, metrics are saved in:

- `outputs/linear_regression/`
- `outputs/random_forest/`

Each run saves a separate JSON file with date and time in the filename.

## Feature importance (random forest regression)
The model is strongly driven by median income, which accounts for more than 50% of total feature importance. 
Geographic features (latitude and longitude) also contribute significantly, confirming the spatial nature of housing prices. 
Engineered ratio-based features provide additional predictive value, while raw room counts appear less informative.