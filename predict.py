from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import smiles_to_features, is_valid_smiles


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submit.csv"
MODEL_PATH = MODEL_DIR / "si_model.joblib"

OUTPUT_PATH = RESULTS_DIR / "submission.csv"


def main():

    print("=" * 55)
    print("ПРЕДСКАЗАНИЕ SI VACV")
    print("=" * 55)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Проверяем наличие файлов
    for path in [TEST_PATH, SAMPLE_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден файл: {path}"
            )

    # Загружаем данные
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    if "SMILES" not in test.columns:
        raise ValueError(
            "В test.csv отсутствует столбец SMILES."
        )

    if "index" not in sample.columns:
        raise ValueError(
            "В sample_submit.csv отсутствует столбец index."
        )

    if "SI VACV" not in sample.columns:
        raise ValueError(
            "В sample_submit.csv отсутствует столбец SI VACV."
        )

    # Проверяем SMILES
    invalid = ~test["SMILES"].apply(is_valid_smiles)

    if invalid.any():
        bad_rows = test.index[invalid].tolist()[:10]
        raise ValueError(
            f"В test.csv найдены некорректные SMILES. "
            f"Первые номера строк: {bad_rows}"
        )

    # Загружаем обученную модель
    print("Загружаем Random Forest...")
    model = joblib.load(MODEL_PATH)

    # Формируем признаки
    cc50_values = (
        test["CC50_uM"]
        if "CC50_uM" in test.columns
        else None
    )

    X = smiles_to_features(
        test["SMILES"],
        cc50_values
    )

    # Получаем предсказания в log10(SI)
    print("Выполняем предсказание...")

    log_predictions = model.predict(X)

    # Возвращаемся к обычной шкале SI
    predictions = np.power(10, log_predictions)

    # Проверяем порядок строк относительно sample submission
    if "index" in test.columns:
        if test["index"].duplicated().any():
            raise ValueError(
                "В test.csv есть повторяющиеся значения index."
            )

        indexed_test = test.set_index("index")

        if not sample["index"].isin(indexed_test.index).all():
            raise ValueError(
                "Не все индексы sample_submit.csv найдены в test.csv."
            )

        prediction_by_index = pd.Series(
            predictions,
            index=test["index"]
        )

        ordered_predictions = sample["index"].map(
            prediction_by_index
        ).to_numpy()

    else:
        if len(test) != len(sample):
            raise ValueError(
                "Количество строк test.csv не совпадает "
                "с количеством строк sample_submit.csv."
            )

        ordered_predictions = predictions

    # Формируем итоговый файл
    submission = pd.DataFrame({
        "index": sample["index"],
        "SI VACV": ordered_predictions
    })

    submission.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nПредсказание завершено!")
    print(f"Количество молекул: {len(submission)}")
    print(f"Файл сохранён: {OUTPUT_PATH}")
    print("\nПервые результаты:")
    print(submission.head())

    print("=" * 55)


if __name__ == "__main__":
    main()