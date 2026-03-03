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
- This is not "overfitting" because the correction is moderate.

### Random Forest Regression
Measured metrics:
- MAE (Mean Average Error): 0.3510
- MSE (Mean Square Error): 0.2870
- RMSE (Root Mean Square Error): 0.5358
- R2 (Coefficient of determination): 0.7809

The substantial performance improvement from Linear Regression (R² ≈ 0.59) to Random Forest (R² ≈ 0.78) indicates that the relationship between features and target is strongly non-linear.
