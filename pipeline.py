"""Полный пайплайн одной командой: обучение + проверка -> results/submission.csv.
Запуск: python pipeline.py [--no-external]"""
from src import train, predict

if __name__ == "__main__":
    train.main()     # scaffold-CV, обучение, models/si_model.joblib
    predict.main()   # предсказание теста, results/submission.csv
