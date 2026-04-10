"""
Задача 5: ClearML интеграция — просмотр экспериментов и управление
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering

st.set_page_config(page_title="ClearML", layout="wide")
st.title("📡 ClearML — Управление экспериментами")

st.markdown("""
ClearML используется для отслеживания всех экспериментов проекта,
версионирования датасетов и логирования метрик.

Скрипты для запуска экспериментов находятся в папке `clearml_scripts/`.
""")

# ── Информация о датасете ─────────────────────────────────────────────────────
st.subheader("🗂️ Датасет в ClearML")
with st.spinner("Загрузка данных..."):
    df   = load_data()
    df_p = feature_engineering(preprocess(df))

target = 'UltimateIncurredClaimCost'
X = df_p.drop(columns=[target])
y = np.log1p(df_p[target])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Записей",   f"{len(df):,}")
col2.metric("Признаков (после FE)", X.shape[1])
col3.metric("Пропусков в исходных данных", f"{df.isnull().sum().sum():,}")
col4.metric("Целевая переменная", "UltimateIncurredClaimCost")

st.markdown("""
**Датасет:** Workers Compensation Dataset (OpenML ID: 42876)
**Проект ClearML:** Workers Compensation
**Задача:** Регрессия — прогнозирование итоговой стоимости страхового возмещения
""")

# ── Запуск экспериментов прямо из UI ─────────────────────────────────────────
st.subheader("🚀 Запуск экспериментов (локально)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if st.button("▶ Запустить Эксперимент 1 — Linear Regression (baseline)",
             type="primary"):
    with st.spinner("Обучение Linear Regression..."):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_tr_s, y_train)
        y_pred_lr = lr.predict(X_te_s)

        r2_lr   = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        mae_lr  = mean_absolute_error(y_test, y_pred_lr)

    st.success("✅ Эксперимент 1 завершён!")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²",   f"{r2_lr:.4f}")
    c2.metric("RMSE", f"{rmse_lr:.4f}")
    c3.metric("MAE",  f"{mae_lr:.4f}")

    st.session_state['exp1'] = {
        "R²": round(r2_lr, 4),
        "RMSE": round(rmse_lr, 4),
        "MAE": round(mae_lr, 4)
    }

    # Scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred_lr, alpha=0.3, s=10, color='steelblue')
    lims = [float(y_test.min()), float(y_test.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Идеальный прогноз')
    ax.set_xlabel("Факт (log)"); ax.set_ylabel("Прогноз (log)")
    ax.set_title("Linear Regression: Факт vs Прогноз")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

if st.button("▶ Запустить Эксперимент 2 — XGBoost (оптимизированный)",
             type="primary"):
    with st.spinner("Обучение XGBoost с лучшими параметрами..."):
        best_params = {
            'n_estimators': 200, 'max_depth': 6,
            'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': 0
        }
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_test)

        r2_xgb   = r2_score(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)

    st.success("✅ Эксперимент 2 завершён!")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²",   f"{r2_xgb:.4f}")
    c2.metric("RMSE", f"{rmse_xgb:.4f}")
    c3.metric("MAE",  f"{mae_xgb:.4f}")

    st.session_state['exp2'] = {
        "R²": round(r2_xgb, 4),
        "RMSE": round(rmse_xgb, 4),
        "MAE": round(mae_xgb, 4)
    }

    # Feature Importance
    fi = pd.Series(model.feature_importances_,
                   index=X.columns).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fi.index[::-1], fi.values[::-1], color='steelblue')
    ax.set_xlabel("Важность")
    ax.set_title("Топ-10 важных признаков (XGBoost)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Сравнительная таблица ─────────────────────────────────────────────────────
if 'exp1' in st.session_state or 'exp2' in st.session_state:
    st.subheader("📊 Результаты экспериментов")
    rows = []
    if 'exp1' in st.session_state:
        rows.append({"Эксперимент": "Experiment 1 — Linear Regression",
                     **st.session_state['exp1'], "Статус": "✅ Завершён"})
    if 'exp2' in st.session_state:
        rows.append({"Эксперимент": "Experiment 2 — XGBoost (Optuna)",
                     **st.session_state['exp2'], "Статус": "✅ Завершён"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("Нажмите кнопки выше для запуска экспериментов.")

# ── Информация о ClearML скриптах ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🔗 ClearML скрипты")
st.markdown("""
Для запуска экспериментов с отслеживанием через ClearML Web:

```bash
# 1. Создать датасет в ClearML
python clearml_scripts/dataset_creation.py

# 2. Запустить эксперимент 1 (Linear Regression baseline)
python clearml_scripts/experiment1.py

# 3. Запустить эксперимент 2 (XGBoost + Optuna)
python clearml_scripts/experiment2.py
```

**ClearML Web:** https://app.clear.ml
**Проект:** Workers Compensation
""")
