"""
Задача 7: Интерпретируемость модели с SHAP
"""
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SHAP", layout="wide")
st.title("💡 Интерпретируемость модели — SHAP")
st.markdown("SHAP (SHapley Additive exPlanations) объясняет вклад "
            "каждого признака в предсказание модели.")

with st.spinner("Загрузка и обучение..."):
    df    = load_data()
    df_p  = feature_engineering(preprocess(df))
    target = 'UltimateIncurredClaimCost'
    X = df_p.drop(columns=[target])
    y = np.log1p(df_p[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(n_estimators=100, random_state=42,
                               verbosity=0)
    model.fit(X_train, y_train)
    explainer   = shap.TreeExplainer(model)
    sample      = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

st.success("✅ SHAP-значения вычислены!")

# ── Summary Plot ────────────────────────────────────────────────────────────
st.subheader("Summary Plot — влияние признаков на предсказания")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, sample, plot_type="bar",
                  show=False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Beeswarm ────────────────────────────────────────────────────────────────
st.subheader("Beeswarm Plot — детальное распределение SHAP-значений")
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Waterfall для одного наблюдения ─────────────────────────────────────────
st.subheader("Waterfall Plot — объяснение одного предсказания")
idx = st.slider("Выберите наблюдение:", 0, len(sample)-1, 0)
fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(values=shap_values[idx],
                     base_values=float(explainer.expected_value),
                     data=sample.iloc[idx].values,
                     feature_names=list(sample.columns)),
    show=False
)
plt.tight_layout()
st.pyplot(fig)
plt.close()