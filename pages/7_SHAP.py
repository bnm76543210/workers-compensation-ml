"""
Задача 7: Интерпретируемость — SHAP + LIME + Partial Dependence Plots
"""
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering

st.set_page_config(page_title="Интерпретируемость", layout="wide")
st.title("Интерпретируемость модели — SHAP, LIME, PDP")

with st.spinner("Загрузка и обучение XGBoost..."):
    df    = load_data()
    df_p  = feature_engineering(preprocess(df))
    target = 'UltimateIncurredClaimCost'
    X = df_p.drop(columns=[target])
    y = np.log1p(df_p[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

st.success("Модель XGBoost обучена!")

tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "Partial Dependence Plots"])

# ── TAB 1: SHAP ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("SHAP — SHapley Additive exPlanations")
    st.markdown("""
    SHAP объясняет вклад каждого признака в каждое отдельное предсказание,
    основываясь на теории игр (значения Шепли).
    """)

    with st.spinner("Вычисление SHAP-значений (200 наблюдений)..."):
        explainer   = shap.TreeExplainer(model)
        sample      = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample)

    st.subheader("Summary Plot — важность признаков (глобальная)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Beeswarm Plot — детальное распределение SHAP-значений")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Waterfall Plot — объяснение одного предсказания")
    idx = st.slider("Выберите наблюдение:", 0, len(sample)-1, 0)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[idx],
                         base_values=float(explainer.expected_value),
                         data=sample.iloc[idx].values,
                         feature_names=list(sample.columns)),
        show=False)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # SHAP значения в таблице
    st.subheader("Таблица SHAP-значений для выбранного наблюдения")
    shap_df = pd.DataFrame({
        'Признак': list(sample.columns),
        'Значение признака': sample.iloc[idx].values.round(4),
        'SHAP-вклад': shap_values[idx].round(4),
    }).sort_values('SHAP-вклад', key=abs, ascending=False)
    st.dataframe(shap_df, use_container_width=True)

# ── TAB 2: LIME ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("LIME — Local Interpretable Model-agnostic Explanations")
    st.markdown("""
    LIME объясняет **одно конкретное предсказание** через локальную
    линейную аппроксимацию в окрестности этого наблюдения.
    В отличие от SHAP, LIME работает с любой моделью (model-agnostic).
    """)

    lime_idx = st.slider("Выберите наблюдение для LIME:", 0,
                          min(499, len(X_test)-1), 0, key="lime_idx")

    if st.button("Объяснить с LIME", type="primary"):
        with st.spinner("LIME вычисляется..."):
            try:
                import lime
                import lime.lime_tabular

                explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=list(X_train.columns),
                    mode='regression',
                    random_state=42)

                exp = explainer_lime.explain_instance(
                    X_test.values[lime_idx],
                    model.predict,
                    num_features=10)

                # Получаем список (признак, вклад)
                lime_vals = exp.as_list()
                lime_df = pd.DataFrame(lime_vals,
                                       columns=['Условие', 'Вклад'])
                lime_df = lime_df.sort_values('Вклад', key=abs, ascending=True)

                # Plotly bar
                colors = ['#4CAF50' if v > 0 else '#F44336'
                          for v in lime_df['Вклад']]
                fig_lime = go.Figure(go.Bar(
                    x=lime_df['Вклад'],
                    y=lime_df['Условие'],
                    orientation='h',
                    marker_color=colors,
                ))
                fig_lime.update_layout(
                    title=f"LIME: объяснение предсказания #{lime_idx}",
                    xaxis_title="Вклад в предсказание (log-шкала)",
                    template='plotly_white',
                    height=400)
                st.plotly_chart(fig_lime, use_container_width=True)

                # Предсказание
                pred_log = model.predict(X_test.values[[lime_idx]])[0]
                pred_usd = np.expm1(pred_log)
                real_usd = np.expm1(y_test.values[lime_idx])
                c1, c2 = st.columns(2)
                c1.metric("Прогноз модели", f"${pred_usd:,.0f}")
                c2.metric("Факт", f"${real_usd:,.0f}")

                st.caption(
                    "Зелёные полосы увеличивают предсказание, "
                    "красные — уменьшают.")

            except Exception as e:
                st.error(f"Ошибка LIME: {e}")

# ── TAB 3: PDP ───────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Partial Dependence Plots (PDP)")
    st.markdown("""
    PDP показывает, как **изменение одного признака** влияет на предсказание модели,
    при усреднении по всем остальным признакам.
    """)

    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    feat1 = st.selectbox("Признак для PDP:", numeric_feats,
                         index=numeric_feats.index('InitialCaseEstimate')
                         if 'InitialCaseEstimate' in numeric_feats else 0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Построить 1D PDP", type="primary", key="pdp1d"):
            with st.spinner("Вычисление PDP..."):
                # Используем небольшую выборку для скорости
                sample_pdp = X_test.sample(min(500, len(X_test)),
                                            random_state=42)
                feat_idx   = list(X.columns).index(feat1)
                feature_vals = np.linspace(
                    float(X[feat1].quantile(0.05)),
                    float(X[feat1].quantile(0.95)), 50)
                avg_preds = []
                for val in feature_vals:
                    X_mod = sample_pdp.copy()
                    X_mod[feat1] = val
                    avg_preds.append(np.expm1(model.predict(X_mod)).mean())

            fig_pdp = go.Figure()
            fig_pdp.add_trace(go.Scatter(
                x=feature_vals, y=avg_preds, mode='lines',
                line=dict(color='steelblue', width=2),
                name='PDP'))
            fig_pdp.update_layout(
                title=f"Partial Dependence Plot: {feat1}",
                xaxis_title=feat1,
                yaxis_title="Среднее предсказание ($)",
                template='plotly_white')
            st.plotly_chart(fig_pdp, use_container_width=True)
            st.caption(
                f"График показывает, как изменяется средняя прогнозируемая "
                f"стоимость выплаты при изменении '{feat1}', "
                f"при прочих равных условиях.")

    with col2:
        feat2 = st.selectbox("Второй признак (для 2D PDP):", numeric_feats,
                              index=numeric_feats.index('WeeklyPay')
                              if 'WeeklyPay' in numeric_feats else 1,
                              key="feat2")
        if st.button("Построить 2D PDP (heatmap)", type="primary", key="pdp2d"):
            with st.spinner("Вычисление 2D PDP..."):
                sample_pdp = X_test.sample(min(300, len(X_test)),
                                            random_state=42)
                v1 = np.linspace(float(X[feat1].quantile(0.1)),
                                  float(X[feat1].quantile(0.9)), 20)
                v2 = np.linspace(float(X[feat2].quantile(0.1)),
                                  float(X[feat2].quantile(0.9)), 20)
                z = np.zeros((len(v2), len(v1)))
                for i, val1 in enumerate(v1):
                    for j, val2 in enumerate(v2):
                        X_mod = sample_pdp.copy()
                        X_mod[feat1] = val1
                        X_mod[feat2] = val2
                        z[j, i] = np.expm1(model.predict(X_mod)).mean()

            fig_2d = go.Figure(go.Heatmap(
                x=v1, y=v2, z=z,
                colorscale='viridis',
                colorbar=dict(title='Прогноз ($)')))
            fig_2d.update_layout(
                title=f"2D PDP: {feat1} x {feat2}",
                xaxis_title=feat1, yaxis_title=feat2,
                template='plotly_white')
            st.plotly_chart(fig_2d, use_container_width=True)
