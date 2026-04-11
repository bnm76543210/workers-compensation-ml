"""
Задача 4: Мощные модели — XGBoost, LightGBM, CatBoost, Stacking
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib, os
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
from src.logger import log
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Модели", layout="wide")
st.title("Обучение и сравнение моделей")
log("=== Страница 4: Модели загружена ===")

with st.spinner("Загрузка данных..."):
    df = load_data()

df_proc = feature_engineering(preprocess(df))
target = 'UltimateIncurredClaimCost'
X = df_proc.drop(columns=[target])
y = np.log1p(df_proc[target])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Реестр моделей ───────────────────────────────────────────────────────────
MODELS = {
    "Linear Regression (baseline)": LinearRegression(),
    "Random Forest":   RandomForestRegressor(n_estimators=100,
                                              random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100,
                                                    random_state=42),
    "XGBoost":         xgb.XGBRegressor(n_estimators=100, random_state=42,
                                         verbosity=0),
    "LightGBM":        lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                          verbose=-1),
    "CatBoost":        CatBoostRegressor(iterations=100, random_seed=42,
                                          verbose=0),
}

ENSEMBLE_MODELS = {
    "Voting Regressor": VotingRegressor(estimators=[
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42,
                                  verbosity=0)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                   verbose=-1)),
        ('rf',  RandomForestRegressor(n_estimators=50, random_state=42,
                                      n_jobs=-1)),
    ]),
    "Stacking": StackingRegressor(
        estimators=[
            ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42,
                                      verbosity=0)),
            ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                       verbose=-1)),
            ('cat', CatBoostRegressor(iterations=100, random_seed=42,
                                       verbose=0)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=-1
    ),
}

tab1, tab2 = st.tabs(["Базовые модели", "Ансамблевые методы"])

# ── TAB 1: Базовые модели ────────────────────────────────────────────────────
with tab1:
    st.markdown("Сравнение 6 базовых моделей: от линейной регрессии до CatBoost.")
    if st.button("Обучить все базовые модели", type="primary", key="train_base"):
        log("Пользователь: нажата кнопка обучения базовых моделей")
        results = {}
        os.makedirs("models", exist_ok=True)
        progress = st.progress(0)

        for i, (name, model) in enumerate(MODELS.items()):
            with st.spinner(f"Обучение {name}..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_orig  = np.expm1(y_pred)
                y_test_orig  = np.expm1(y_test)
                results[name] = {
                    "R2":      round(r2_score(y_test, y_pred), 4),
                    "RMSE_log": round(float(np.sqrt(
                        mean_squared_error(y_test, y_pred))), 4),
                    "MAE_USD":  round(float(
                        mean_absolute_error(y_test_orig, y_pred_orig)), 0),
                }
                joblib.dump(model,
                    f"models/{name.replace(' ','_').replace('(','').replace(')','')}.pkl")
            progress.progress((i+1) / len(MODELS))

        log("Базовые модели обучены",
            **{n: f"R2={v['R2']} MAE=${v['MAE_USD']:,.0f}" for n, v in results.items()})
        st.success("Все модели обучены!")
        st.session_state['base_results'] = results
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['models'] = MODELS

    if 'base_results' in st.session_state:
        res_df = pd.DataFrame(st.session_state['base_results']).T
        res_df.index.name = 'Модель'

        st.subheader("Сравнение метрик")
        st.dataframe(
            res_df.style
                  .highlight_max(subset=['R2'], color='lightgreen')
                  .highlight_min(subset=['RMSE_log', 'MAE_USD'],
                                 color='lightgreen')
        )

        # Plotly bar
        fig = go.Figure()
        models_list = list(st.session_state['base_results'].keys())
        for metric, col in [('R2','steelblue'), ('RMSE_log','#E57373'),
                             ('MAE_USD','#81C784')]:
            vals = [st.session_state['base_results'][m][metric]
                    for m in models_list]
            fig.add_trace(go.Bar(name=metric, x=models_list, y=vals,
                                 marker_color=col))
        fig.update_layout(barmode='group', template='plotly_white',
                          title='Сравнение моделей по метрикам',
                          xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

        # Лучшая модель: Predicted vs Actual
        best_name = max(st.session_state['base_results'],
                        key=lambda x: st.session_state['base_results'][x]['R2'])
        st.markdown(f"**Лучшая модель:** {best_name} "
                    f"(R2={st.session_state['base_results'][best_name]['R2']})")

        best_model = st.session_state['models'][best_name]
        y_pred_best = best_model.predict(st.session_state['X_test'])
        y_test_orig = np.expm1(st.session_state['y_test'])
        y_pred_orig = np.expm1(y_pred_best)
        sample_idx  = np.random.RandomState(42).choice(
            len(y_test_orig), min(2000, len(y_test_orig)), replace=False)
        fig_sc = px.scatter(x=y_test_orig.values[sample_idx],
                            y=y_pred_orig[sample_idx],
                            opacity=0.3,
                            title=f"{best_name}: Факт vs Прогноз (USD)",
                            labels={'x': 'Факт ($)', 'y': 'Прогноз ($)'},
                            template='plotly_white')
        max_v = float(max(y_test_orig.max(), y_pred_orig.max()))
        fig_sc.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines',
                                    line=dict(color='red', dash='dash'),
                                    name='Идеал'))
        st.plotly_chart(fig_sc, use_container_width=True)

# ── TAB 2: Ансамблевые методы ────────────────────────────────────────────────
with tab2:
    st.markdown("**Voting Regressor** — усредняет предсказания XGBoost + LightGBM + RF.")
    st.markdown("**Stacking** — мета-модель (Ridge) обучается на предсказаниях "
                "XGBoost + LightGBM + CatBoost.")
    st.warning("Обучение ансамблей занимает 2-5 минут.")

    if st.button("Обучить ансамблевые модели", type="primary", key="train_ensemble"):
        log("Пользователь: нажата кнопка обучения ансамблей")
        ens_results = {}
        progress2 = st.progress(0)

        for i, (name, model) in enumerate(ENSEMBLE_MODELS.items()):
            with st.spinner(f"Обучение {name}..."):
                model.fit(X_train, y_train)
                y_pred     = model.predict(X_test)
                y_pred_orig = np.expm1(y_pred)
                y_test_orig = np.expm1(y_test)
                ens_results[name] = {
                    "R2":       round(r2_score(y_test, y_pred), 4),
                    "RMSE_log": round(float(np.sqrt(
                        mean_squared_error(y_test, y_pred))), 4),
                    "MAE_USD":  round(float(
                        mean_absolute_error(y_test_orig, y_pred_orig)), 0),
                }
                joblib.dump(model, f"models/{name.replace(' ','_')}.pkl")
            progress2.progress((i+1) / len(ENSEMBLE_MODELS))

        log("Ансамблевые модели обучены",
            **{n: f"R2={v['R2']} MAE=${v['MAE_USD']:,.0f}" for n, v in ens_results.items()})
        st.success("Ансамблевые модели обучены!")
        st.session_state['ens_results'] = ens_results
        st.session_state['ens_models']  = ENSEMBLE_MODELS

    if 'ens_results' in st.session_state:
        ens_df = pd.DataFrame(st.session_state['ens_results']).T
        ens_df.index.name = 'Модель'
        st.dataframe(
            ens_df.style
                  .highlight_max(subset=['R2'], color='lightgreen')
                  .highlight_min(subset=['RMSE_log', 'MAE_USD'],
                                 color='lightgreen')
        )

        # Сравнение с базовыми
        if 'base_results' in st.session_state:
            st.subheader("Сравнение базовых vs ансамблевых")
            all_r = {**st.session_state['base_results'],
                     **st.session_state['ens_results']}
            all_df = pd.DataFrame(all_r).T[['R2', 'MAE_USD']]
            all_df = all_df.sort_values('R2', ascending=False)
            fig_all = px.bar(all_df, x=all_df.index, y='R2',
                             title='R2 всех моделей (выше = лучше)',
                             template='plotly_white',
                             color='R2', color_continuous_scale='greens',
                             labels={'x': 'Модель'})
            fig_all.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_all, use_container_width=True)
