
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_model():

    model = RandomForestRegressor(
        n_estimators=500,
        max_features=0.7,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("random_forest", model)
    ])

    return pipeline