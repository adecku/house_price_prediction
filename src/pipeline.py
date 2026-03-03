from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import add_features


def build_pipeline(model=None) -> Pipeline:
    if model is None:
        model = LinearRegression()

    feature_engineering = FunctionTransformer(add_features, validate=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )
