"""
Задача 6: Анализ ошибок модели
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering

st.set_page_config(page_title="Анализ ошибок", layout="wide")
st.title("🔍 Анализ ошибок модели")

with st.spinner("Загрузка и обучение..."):
    df   = load_data()
    df_p = feature_engineering(preprocess(df))
    target = 'UltimateIncurredClaimCost'
    X = df_p.drop(columns=[target])
    y = np.log1p(df_p[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(n_estimators=100, random_state=42,
                               verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Возвращаем в исходный масштаб
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred)

errors_df = X_test.copy()
errors_df['Фактическая_стоимость']    = y_test_orig.values
errors_df['Предсказанная_стоимость']  = y_pred_orig
errors_df['Абс_ошибка']  = np.abs(y_test_orig.values - y_pred_orig)
errors_df['Отн_ошибка']  = errors_df['Абс_ошибка'] / \
                            (y_test_orig.values + 1) * 100
errors_df = errors_df.sort_values('Абс_ошибка', ascending=False)

# ── Топ ошибок ──────────────────────────────────────────────────────────────
st.subheader("Топ-20 случаев с наибольшей ошибкой")
st.dataframe(errors_df[['Фактическая_стоимость',
                         'Предсказанная_стоимость',
                         'Абс_ошибка',
                         'Отн_ошибка']].head(20).round(2))

# ── Распределение ошибок ────────────────────────────────────────────────────
st.subheader("Распределение абсолютных ошибок")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(errors_df['Абс_ошибка'], bins=50,
             color='steelblue', edgecolor='white')
axes[0].set_title("Распределение абс. ошибок")
axes[0].set_xlabel("Ошибка ($)")

axes[1].scatter(y_test_orig, y_pred_orig,
                alpha=0.3, color='steelblue', s=10)
lims = [0, max(y_test_orig.max(), y_pred_orig.max())]
axes[1].plot(lims, lims, 'r--', linewidth=1.5)
axes[1].set_xlabel("Факт ($)")
axes[1].set_ylabel("Прогноз ($)")
axes[1].set_title("Факт vs Прогноз")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Метрики ─────────────────────────────────────────────────────────────────
st.subheader("Метрики качества")
m1, m2, m3 = st.columns(3)
m1.metric("R²",   f"{r2_score(y_test, y_pred):.4f}")
m2.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)):,.0f}")
m3.metric("MAE",  f"${errors_df['Абс_ошибка'].mean():,.0f}")