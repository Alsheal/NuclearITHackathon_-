from lightgbm import LGBMRegressor


def make_model():
    """Единственная модель проекта: LightGBM (обучается на остатке от базиса)."""
    return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8,
                         min_child_samples=5, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.3, reg_lambda=5, random_state=42, verbose=-1)
