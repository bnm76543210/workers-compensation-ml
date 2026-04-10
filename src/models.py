"""
Централизованное хранилище моделей и функций обучения/оценки.
Используется страницами pages/3_Models.py и pages/4_Optimization.py
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── Реестр доступных моделей ────────────────────────────────────────────────
def get_models() -> dict:
    """Возвращает словарь всех доступных моделей с параметрами по умолчанию."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=100,
                                 random_state=42,
                                 n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=100,
                                 random_state=42),
        "XGBoost":           xgb.XGBRegressor(
                                 n_estimators=100,
                                 random_state=42,
                                 verbosity=0),
        "LightGBM":          lgb.LGBMRegressor(
                                 n_estimators=100,
                                 random_state=42,
                                 verbose=-1),
    }

# ── Подготовка данных ───────────────────────────────────────────────────────
def prepare_xy(df: pd.DataFrame):
    """
    Разделяет DataFrame на X и y.
    Применяет log1p к целевой переменной для нормализации распределения.
    """
    target = 'UltimateIncurredClaimCost'
    X = df.drop(columns=[target])
    y = np.log1p(df[target])
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42):
    """Разделяет данные на train/test."""
    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state)

# ── Обучение одной модели ───────────────────────────────────────────────────
def train_single(model, X_train, y_train):
    """Обучает модель и возвращает её."""
    model.fit(X_train, y_train)
    return model

# ── Оценка модели ───────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    """
    Вычисляет метрики качества модели.
    Возвращает словарь с R², RMSE, MAE.
    """
    y_pred = model.predict(X_test)
    return {
        "R²":   round(r2_score(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
    }

def cross_validate(model, X, y, cv: int = 5) -> dict:
    """
    Выполняет K-Fold кросс-валидацию.
    Возвращает среднее и стандартное отклонение R².
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    return {
        "CV R² mean": round(scores.mean(), 4),
        "CV R² std":  round(scores.std(), 4),
        "CV scores":  scores.tolist(),
    }

# ── Сохранение / загрузка модели ────────────────────────────────────────────
def save_model(model, name: str, folder: str = "models"):
    """Сохраняет модель в папку models/."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name.replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    return path

def load_model(name: str, folder: str = "models"):
    """Загружает модель из папки models/."""
    path = os.path.join(folder, f"{name.replace(' ', '_')}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return joblib.load(path)

# ── Обучение всех моделей и сравнение ───────────────────────────────────────
def train_and_compare(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Обучает все модели из реестра и возвращает таблицу сравнения метрик.
    Также сохраняет каждую модель в models/.
    """
    models  = get_models()
    results = {}

    for name, model in models.items():
        trained = train_single(model, X_train, y_train)
        metrics = evaluate(trained, X_test, y_test)
        results[name] = metrics
        save_model(trained, name)

    return pd.DataFrame(results).T