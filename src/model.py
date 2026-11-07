from lightgbm import LGBMRegressor

def make_model():
    return LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=8,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.3,
        reg_lambda=3.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )