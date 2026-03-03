# DEVLOG to track progress

### Data analisis
Conslusions:
- **MedInc** highly correlates with **MedHouseVal**
- The target distribution can be asymmetric and has an upper limit (values ​​at the limit), which may make it difficult to predict extremely expensive properties.
- There are no missing data records. 
- There are outliers in few features ('AveRooms', 'Population', 'AveOccup').

### Basic linear regression
Measured metrics:
- MAE (Mean Average Error): 0.5332
- MSE (Mean Square Error): 0.5559
- RMSE (Root Mean Square Error): 0.7456
- R2 (Coefficient of determination): 0.5758

Conslusions:
- MSE: the model is off by 53% of the full range
- RMSE: large prediction errors exist, model struggles on certain regions
- R^2: the model explains 57,6% of the housing price variation, the rest remains unexplained


### Add ratio-features
Added features:
- rooms_per_household
- bedrooms_ratio (% of all rooms)
- population_per_household

### Basic linear regression with added ratio-features
Measured metrics:
- MAE (Mean Average Error): 0.5261
- MSE (Mean Square Error): 0.5317
- RMSE (Root Mean Square Error): 0.7292
- R2 (Coefficient of determination): 0.5942

Conslusions:
- MAE, MSE and RMSE went down, whereas R2 went up
- This is not overfitting because the correction is moderate.

### Random Forest Regression
Measured metrics:
- MAE (Mean Average Error): 0.3510
- MSE (Mean Square Error): 0.2870
- RMSE (Root Mean Square Error): 0.5358
- R2 (Coefficient of determination): 0.7809

The substantial performance improvement from Linear Regression (R² ≈ 0.59) to Random Forest (R² ≈ 0.78) indicates that the relationship between features and target is strongly non-linear.

### Random Forest Regression - Cross Variation
Measured metrics:
- MAE (Mean Average Error): 0.3555 +- 0.0072
- MSE (Mean Square Error): 0.2891 +- 0.0083
- RMSE (Root Mean Square Error): 0.5376 +- 0.0077
- R2 (Coefficient of determination): 0.7836 +- 0.0082

Achieved R² ≈ 0.78 with low cross-validation variance (±0.008), demonstrating strong generalization and superior modeling of non-linear feature interactions.

### Random Forest Regression - Finetuning
Used following values of parameters:
param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
    }

Best combination:
"best_params": {
    "model__n_estimators": 300,
    "model__min_samples_split": 5,
    "model__min_samples_leaf": 1,
    "model__max_features": 0.8,
    "model__max_depth": null
  }

Best measured metrics:
- MAE (Mean Average Error): 0.3316
- MSE (Mean Square Error): 0.2580
- RMSE (Root Mean Square Error): 0.5080
- R2 (Coefficient of determination): 0.8030

There is no observed over or underfitting, good balance bias-variance. The model seems to be well tuned - close to the maximum. 

### Add feature importance for random forest regression
Results are in '/outputs/random_forest/'.

Conslusion:
The model is strongly driven by median income, which accounts for more than 50% of total feature importance. Geographic features (latitude and longitude) also contribute significantly, confirming the spatial nature of housing prices. Engineered ratio-based features provide additional predictive value, while raw room counts appear less informative.