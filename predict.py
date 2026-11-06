import argparse
import os

import joblib
import pandas as pd

from src.si_model import predict_si, predict_si_log


def main():

    parser = argparse.ArgumentParser(
        description="Предсказание индекса селективности SI"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV-файл с молекулами"
    )

    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Файл для сохранения результатов"
    )

    args = parser.parse_args()

    model_path = "models/si_model.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Модель не найдена. Сначала запустите train.py"
        )

    print("Загрузка модели...")

    model_package = joblib.load(model_path)

    print("Загрузка молекул...")

    data = pd.read_csv(args.input)

    if "smiles" not in data.columns:
        raise ValueError(
            "В исходном файле должен быть столбец smiles"
        )

    # Предсказываем SI
    predicted_si = predict_si(
        data["smiles"].tolist(),
        model_package
    )

    predicted_log_si = predict_si_log(
        data["smiles"].tolist(),
        model_package
    )

    data["SI_predicted"] = predicted_si
    data["log10_SI_predicted"] = predicted_log_si

    data.to_csv(
        args.output,
        index=False
    )

    print("Предсказание завершено!")
    print(f"Результат сохранён: {args.output}")


if __name__ == "__main__":
    main()