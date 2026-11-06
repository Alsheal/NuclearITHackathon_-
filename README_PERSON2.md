# Участник №2 — модель машинного обучения

## Назначение

Разработка модели машинного обучения для предсказания
индекса селективности SI химических соединений
против вируса VACV.

## Формула

SI = CC50 / IC50

- IC50 — концентрация, подавляющая 50% активности вируса.
- CC50 — концентрация, вызывающая гибель 50% клеток.
- SI — индекс селективности.

## Используемые технологии

- Python
- RDKit
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Признаки молекул

1. Morgan Fingerprints
2. Molecular Weight
3. LogP
4. TPSA
5. Ring Count
6. NumHDonors
7. NumHAcceptors

## Алгоритмы

- Random Forest Regressor
- Extra Trees Regressor

Алгоритмы сравниваются с помощью кросс-валидации.
Выбирается модель с наименьшей средней MAE.

## Подготовка данных

Файл data/processed_data.csv должен содержать:

- smiles
- IC50
- CC50

IC50 и CC50 должны быть положительными числами
в одинаковых единицах измерения.

SI рассчитывается программой автоматически.

## Установка

pip install -r requirements.txt

## Обучение

python train.py

## Предсказание

python predict.py --input data/new_compounds.csv --output results/predictions.csv

## Результаты

После обучения создаются:

- models/si_model.joblib
- results/training_report.json
- results/test_predictions.csv

## Ограничения

Модель является исследовательским прототипом.
Предсказания не заменяют лабораторные исследования.

Качество модели зависит от объёма, качества
и сопоставимости экспериментальных данных.