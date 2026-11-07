from lightgbm import LGBMRegressor


def make_model():
    return LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.012,
        num_leaves=63,
        max_depth=8,
        min_child_samples=4,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.45,
        reg_alpha=0.8,
        reg_lambda=5.0,
        min_split_gain=0.01,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )