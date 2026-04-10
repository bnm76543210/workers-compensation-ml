"""
Задача 5: Оптимизация гиперпараметров с Optuna
"""
import streamlit as st
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(page_title="Оптимизация", layout="wide")
st.title("⚙️ Оптимизация гиперпараметров с Optuna")

with st.spinner("Загрузка данных..."):
    df = load_data()
df_proc = feature_engineering(preprocess(df))
target = 'UltimateIncurredClaimCost'
X = df_proc.drop(columns=[target])
y = np.log1p(df_proc[target])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_trials = st.slider("Количество итераций Optuna:", 10, 100, 30, 10)

if st.button("🔍 Запустить оптимизацию", type="primary"):
    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 50, 300),
            'max_depth':       trial.suggest_int('max_depth', 3, 10),
            'learning_rate':   trial.suggest_float('learning_rate',
                                                    0.01, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree',
                                                    0.6, 1.0),
            'random_state': 42, 'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=3, scoring='r2')
        return scores.mean()

    with st.spinner(f"Запуск {n_trials} итераций Optuna..."):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    st.success(f"✅ Оптимизация завершена! Лучший R²: "
               f"{study.best_value:.4f}")

    st.subheader("Лучшие гиперпараметры")
    st.json(best)

    # Обучаем финальную модель с лучшими параметрами
    best_model = xgb.XGBRegressor(**best, random_state=42, verbosity=0)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    r2_opt  = r2_score(y_test, y_pred)
    rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred))

    # Сравнение до/после
    st.subheader("📊 До и после оптимизации")
    base_model = xgb.XGBRegressor(n_estimators=100, random_state=42,
                                   verbosity=0)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    r2_base   = r2_score(y_test, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

    comp = pd.DataFrame({
        "Модель":  ["XGBoost (базовый)", "XGBoost (Optuna)"],
        "R²":      [round(r2_base,4), round(r2_opt,4)],
        "RMSE":    [round(rmse_base,4), round(rmse_opt,4)]
    })
    st.dataframe(comp)

    # График истории оптимизации
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = [t.value for t in study.trials]
    ax.plot(vals, color='steelblue', linewidth=1.5)
    ax.axhline(study.best_value, color='red',
               linestyle='--', label=f'Лучший R²={study.best_value:.4f}')
    ax.set_xlabel("Итерация")
    ax.set_ylabel("R²")
    ax.set_title("История оптимизации Optuna")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()