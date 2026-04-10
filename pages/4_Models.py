"""
Задача 4: Мощные модели — XGBoost, LightGBM, RandomForest
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering

st.set_page_config(page_title="Модели", layout="wide")
st.title("🤖 Обучение и сравнение моделей")

with st.spinner("Загрузка данных..."):
    df = load_data()

df_proc = preprocess(df)
df_proc = feature_engineering(df_proc)

target = 'UltimateIncurredClaimCost'
X = df_proc.drop(columns=[target])
y = np.log1p(df_proc[target])   # log-преобразование целевой переменной

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

MODELS = {
    "Random Forest":       RandomForestRegressor(n_estimators=100,
                                                  random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100,
                                                      random_state=42),
    "XGBoost":             xgb.XGBRegressor(n_estimators=100,
                                             random_state=42,
                                             verbosity=0),
    "LightGBM":            lgb.LGBMRegressor(n_estimators=100,
                                              random_state=42,
                                              verbose=-1),
}

if st.button("🚀 Обучить все модели", type="primary"):
    results = {}
    os.makedirs("models", exist_ok=True)
    progress = st.progress(0)

    for i, (name, model) in enumerate(MODELS.items()):
        with st.spinner(f"Обучение {name}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2   = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            results[name] = {"R²": round(r2,4),
                             "RMSE": round(rmse,4),
                             "MAE":  round(mae,4)}
            joblib.dump(model, f"models/{name.replace(' ','_')}.pkl")
        progress.progress((i+1)/len(MODELS))

    st.success("✅ Все модели обучены!")
    st.session_state['results'] = results
    st.session_state['X_test']  = X_test
    st.session_state['y_test']  = y_test
    st.session_state['models']  = MODELS

if 'results' in st.session_state:
    st.subheader("📊 Сравнение моделей")
    res_df = pd.DataFrame(st.session_state['results']).T
    st.dataframe(res_df.style.highlight_max(subset=['R²'], color='lightgreen')
                             .highlight_min(subset=['RMSE','MAE'],
                                            color='lightgreen'))

    # График сравнения
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, metric in enumerate(['R²', 'RMSE', 'MAE']):
        vals = [st.session_state['results'][m][metric]
                for m in MODELS.keys()]
        axes[idx].bar(list(MODELS.keys()), vals, color='steelblue')
        axes[idx].set_title(metric)
        axes[idx].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()