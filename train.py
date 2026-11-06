import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.si_model import smiles_to_features


# Пути к файлам
DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/si_model.joblib"
REPORT_PATH = "results/training_report.json"
TEST_PATH = "results/test_predictions.csv"


def main():

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Загрузка данных...")

    data = pd.read_csv(DATA_PATH)

    # Проверяем наличие необходимых столбцов
    required_columns = ["smiles", "IC50", "CC50"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"В таблице отсутствует столбец: {column}"
            )

    # Приводим экспериментальные значения к числам
    data["IC50"] = pd.to_numeric(
        data["IC50"],
        errors="coerce"
    )

    data["CC50"] = pd.to_numeric(
        data["CC50"],
        errors="coerce"
    )

    # Удаляем строки без необходимых значений
    data = data.dropna(
        subset=["smiles", "IC50", "CC50"]
    ).copy()

    # Для расчёта SI значения должны быть положительными
    data = data[
        (data["IC50"] > 0) &
        (data["CC50"] > 0)
    ].copy()

    # SI = CC50 / IC50
    data["SI"] = data["CC50"] / data["IC50"]

    # Убираем бесконечные и некорректные значения
    data = data[
        np.isfinite(data["SI"]) &
        (data["SI"] > 0)
    ].copy()

    # Удаляем дубликаты SMILES
    data = data.drop_duplicates(
        subset=["smiles"]
    )

    if len(data) < 10:
        raise ValueError(
            "Недостаточно данных. Нужно минимум 10 корректных строк."
        )

    print(f"Количество молекул: {len(data)}")

    # Преобразуем SMILES в числовые признаки
    print("Преобразование молекул...")

    X = smiles_to_features(
        data["smiles"].tolist()
    )

    # Обучаемся на log10(SI)
    y = np.log10(data["SI"].to_numpy())

    # Делим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = (
        train_test_split(
            X,
            y,
            data["smiles"].to_numpy(),
            test_size=0.2,
            random_state=42
        )
    )

    print("Сравнение алгоритмов...")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    }

    # Кросс-валидация
    number_of_folds = min(5, len(y_train))

    if number_of_folds < 2:
        raise ValueError(
            "Недостаточно обучающих данных для кросс-валидации."
        )

    cv = KFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=42
    )

    results = {}

    best_model = None
    best_model_name = None
    best_score = float("inf")

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        mean_mae = -scores.mean()

        results[name] = {
            "cross_validation_mae_log10": float(mean_mae)
        }

        print(
            f"{name}: MAE = {mean_mae:.4f}"
        )

        if mean_mae < best_score:
            best_score = mean_mae
            best_model = model
            best_model_name = name

    print(f"\nВыбран алгоритм: {best_model_name}")

    # Обучение лучшей модели
    print("Обучение модели...")

    best_model.fit(X_train, y_train)

    # Проверка на тестовой выборке
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nРезультаты проверки:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # Сохраняем модель и параметры
    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "radius": 2,
        "fp_size": 2048,
        "target": "log10(SI)",
        "formula": "SI = CC50 / IC50"
    }

    joblib.dump(
        model_package,
        MODEL_PATH
    )

    # Сохраняем реальные и предсказанные значения
    test_results = pd.DataFrame({
        "smiles": smiles_test,
        "log10_SI_true": y_test,
        "log10_SI_predicted": predictions,
        "SI_true": np.power(10, y_test),
        "SI_predicted": np.power(10, predictions)
    })

    test_results.to_csv(
        TEST_PATH,
        index=False
    )

    # Отчёт
    report = {
        "number_of_compounds": int(len(data)),
        "selected_model": best_model_name,
        "cross_validation_results": results,
        "test_metrics": {
            "MAE_log10": float(mae),
            "RMSE_log10": float(rmse),
            "R2_log10": float(r2)
        },
        "target": "log10(SI)",
        "formula": "SI = CC50 / IC50"
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

    print("\nОбучение завершено!")
    print(f"Модель: {MODEL_PATH}")
    print(f"Отчёт: {REPORT_PATH}")
    print(f"Тестовые результаты: {TEST_PATH}")


if __name__ == "__main__":
    main()