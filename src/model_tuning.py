from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from config import RANDOM_STATE
from pipeline import build_pipeline


def tune_random_forest(X_train, y_train):
    model = build_pipeline(RandomForestRegressor(random_state=RANDOM_STATE))

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_
