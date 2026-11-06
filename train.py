import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from src.features import smiles_to_features
from src.models import get_models


# Пути
DATA_PATH = "data/train.csv"
MODEL_PATH = "models/best_model.pkl"
RESULT_PATH = "results/model_comparison.csv"

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


# 1. Загружаем датасет
df = pd.read_csv(DATA_PATH)

print("Размер датасета:", df.shape)
print("Колонки:", df.columns.tolist())


# 2. Названия колонок
SMILES_COLUMN = "SMILES"
TARGET_COLUMN = "SI VACV"


# 3. Проверяем данные
if SMILES_COLUMN not in df.columns:
    raise ValueError("Не найдена колонка SMILES")

if TARGET_COLUMN not in df.columns:
    raise ValueError("Не найдена колонка SI VACV")


# 4. Убираем отсутствующие значения
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

df = df.dropna(
    subset=[SMILES_COLUMN, TARGET_COLUMN]
).copy()

if len(df) < 10:
    raise ValueError("Недостаточно данных для обучения")


# 5. Убираем дубликаты SMILES
df[SMILES_COLUMN] = df[SMILES_COLUMN].astype(str)

# 6. Создаём признаки только из SMILES
X = smiles_to_features(
    df[SMILES_COLUMN].tolist()
)

y = df[TARGET_COLUMN].to_numpy(
    dtype=np.float64
)

print("Размер матрицы признаков:", X.shape)


# 7. Получаем модели
models = get_models()


# 8. Кросс-валидация
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

for name, model in models.items():

    fold_mse = []

    for train_idx, val_idx in cv.split(X):

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        mse = mean_squared_error(
            y_val,
            predictions
        )

        fold_mse.append(mse)

    results[name] = {
        "MSE_mean": np.mean(fold_mse),
        "MSE_std": np.std(fold_mse)
    }

    print(
        f"{name}: "
        f"MSE = {np.mean(fold_mse):.5f}"
    )


# 9. Сохраняем результаты сравнения
results_df = pd.DataFrame(results).T

results_df.to_csv(
    RESULT_PATH,
    index=True,
    index_label="model"
)


# 10. Выбираем лучшую модель
best_model_name = min(
    results,
    key=lambda name: results[name]["MSE_mean"]
)

print("\nЛучшая модель:", best_model_name)


# 11. Повторно обучаем её на всех данных
best_model = models[best_model_name]

best_model.fit(X, y)


# 12. Сохраняем обученную модель
package = {
    "model": best_model,
    "model_name": best_model_name,
    "target": TARGET_COLUMN,
    "features": "Morgan Fingerprint + molecular descriptors",
    "fp_size": 2048,
    "fp_radius": 2,
    "random_state": 42
}

joblib.dump(
    package,
    MODEL_PATH
)


print("\nОбучение завершено!")
print("Лучшая модель:", best_model_name)
print("Модель сохранена:", MODEL_PATH)
print("Результаты:", RESULT_PATH)