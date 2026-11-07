
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.features import smiles_to_features, is_valid_smiles
from src.models import build_model


# Пути проекта
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TRAIN_PATH = DATA_DIR / "train.csv"
MODEL_PATH = MODEL_DIR / "si_model.joblib"
REPORT_PATH = RESULTS_DIR / "training_report.json"


def main():

    print("=" * 55)
    print("ОБУЧЕНИЕ МОДЕЛИ RANDOM FOREST")
    print("=" * 55)

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Загружаем обучающую таблицу
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл: {TRAIN_PATH}"
        )

    data = pd.read_csv(TRAIN_PATH)

    required_columns = [
        "SMILES",
        "SI VACV",
        "CC50_uM"
    ]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"В train.csv отсутствует столбец: {column}"
            )

    print(f"Всего строк в обучающей таблице: {len(data)}")

    # 2. Оставляем строки с корректными SMILES
    data = data[
        data["SMILES"].apply(is_valid_smiles)
    ].copy()

    # 3. Убираем строки без положительного значения SI
    data["SI VACV"] = pd.to_numeric(
        data["SI VACV"],
        errors="coerce"
    )

    data = data[
        data["SI VACV"].notna()
        & (data["SI VACV"] > 0)
    ].copy()

    if len(data) < 10:
        raise ValueError(
            "Недостаточно корректных обучающих примеров."
        )

    print(f"Примеров после очистки: {len(data)}")

    # 4. Подготавливаем признаки
    X = smiles_to_features(
        data["SMILES"],
        data["CC50_uM"]
    )

    # Модель предсказывает log10(SI)
    y = np.log10(data["SI VACV"].to_numpy())

    print(f"Количество признаков: {X.shape[1]}")

    # 5. Делим данные на обучение и проверку
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Обучающих примеров: {len(X_train)}")
    print(f"Проверочных примеров: {len(X_valid)}")

    # 6. Создаём и обучаем Random Forest
    print("\nНачинаем обучение Random Forest...")

    model = build_model()

    model.fit(X_train, y_train)

    print("Обучение завершено!")

    # 7. Проверяем качество модели
    predictions = model.predict(X_valid)

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_valid, predictions)
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    print("\nРЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("-" * 40)
    print(f"MAE (log10 SI): {mae:.4f}")
    print(f"RMSE (log10 SI): {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # 8. Переобучаем модель на всех доступных данных
    print("\nОбучаем итоговую модель на всех данных...")

    final_model = build_model()
    final_model.fit(X, y)

    # 9. Сохраняем обученную модель
    import joblib

    joblib.dump(final_model, MODEL_PATH)

    # 10. Сохраняем отчёт
    report = {
        "model": "RandomForestRegressor",
        "training_samples": int(len(X)),
        "features": int(X.shape[1]),
        "target": "log10(SI VACV)",
        "validation_MAE_log10": float(mae),
        "validation_RMSE_log10": float(rmse),
        "validation_R2": float(r2),
        "parameters": {
            "n_estimators": 500,
            "max_features": 0.7,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "random_state": 42
        }
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 55)
    print("ГОТОВО!")
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Отчёт сохранён: {REPORT_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()