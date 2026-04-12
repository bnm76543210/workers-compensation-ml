"""
Задача 8: Детальный анализ ошибок модели по сегментам
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
from src.logger import log
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Анализ ошибок", layout="wide")
st.title("Анализ ошибок модели")
log("=== Страница 8: Анализ ошибок загружена ===")

with st.spinner("Загрузка и обучение..."):
    df   = load_data()
    df_p = feature_engineering(preprocess(df))
    target = 'UltimateIncurredClaimCost'
    X = df_p.drop(columns=[target])
    y = np.log1p(df_p[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred)

errors_df = X_test.copy()
errors_df['Фактическая ($)']    = y_test_orig.values
errors_df['Предсказанная ($)']  = y_pred_orig
errors_df['Абс_ошибка ($)']     = np.abs(y_test_orig.values - y_pred_orig)
errors_df['Отн_ошибка (%)']     = (errors_df['Абс_ошибка ($)'] /
                                    (y_test_orig.values + 1) * 100)
errors_df['Ошибка_лог']         = np.abs(y_test.values - y_pred)
errors_df = errors_df.sort_values('Абс_ошибка ($)', ascending=False)

# ── Метрики ──────────────────────────────────────────────────────────────────
st.subheader("Общие метрики качества")
m1, m2, m3, m4 = st.columns(4)
r2_val   = r2_score(y_test, y_pred)
rmse_val = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae_val  = errors_df['Абс_ошибка ($)'].mean()
log("XGBoost обучен для анализа ошибок",
    R2=round(r2_val,4), RMSE_log=round(rmse_val,4), MAE_USD=round(mae_val,0))
m1.metric("R2",   f"{r2_val:.4f}")
m2.metric("RMSE (log)", f"{rmse_val:.4f}")
m3.metric("MAE ($)", f"${mae_val:,.0f}")
m4.metric("Медиана ошибки ($)", f"${errors_df['Абс_ошибка ($)'].median():,.0f}")

# ── Топ ошибок ───────────────────────────────────────────────────────────────
st.subheader("Топ-20 случаев с наибольшей ошибкой")
st.dataframe(errors_df[['Фактическая ($)', 'Предсказанная ($)',
                          'Абс_ошибка ($)', 'Отн_ошибка (%)']
                        ].head(20).round(2))

# ── Распределение ошибок ─────────────────────────────────────────────────────
st.subheader("Распределение ошибок")
col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(errors_df, x='Абс_ошибка ($)', nbins=60,
                            title="Распределение абс. ошибок (USD)",
                            template='plotly_white',
                            color_discrete_sequence=['steelblue'])
    fig_hist.add_vline(x=errors_df['Абс_ошибка ($)'].mean(),
                       line_dash='dash', line_color='red',
                       annotation_text="Среднее")
    st.plotly_chart(fig_hist, width='stretch')

with col2:
    sample_idx = np.random.RandomState(42).choice(
        len(y_test_orig), min(3000, len(y_test_orig)), replace=False)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=y_test_orig.values[sample_idx],
        y=y_pred_orig[sample_idx],
        mode='markers', opacity=0.3,
        marker=dict(size=4, color='steelblue'), name='Предсказания'))
    max_v = float(max(y_test_orig.max(), y_pred_orig.max()))
    fig_sc.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Идеал'))
    fig_sc.update_layout(title="Факт vs Прогноз ($)",
                         xaxis_title="Факт ($)", yaxis_title="Прогноз ($)",
                         template='plotly_white', xaxis_type='log',
                         yaxis_type='log')
    st.plotly_chart(fig_sc, width='stretch')

# ── Анализ ошибок по ценовым сегментам ──────────────────────────────────────
st.subheader("Анализ ошибок по ценовым сегментам")
bins = [0, 500, 2000, 10000, 50000, float('inf')]
labels = ['<$500', '$500-2K', '$2K-10K', '$10K-50K', '>$50K']
errors_df['Сегмент'] = pd.cut(errors_df['Фактическая ($)'],
                               bins=bins, labels=labels)

seg_stats = errors_df.groupby('Сегмент', observed=True).agg(
    Записей=('Абс_ошибка ($)', 'count'),
    MAE=('Абс_ошибка ($)', 'mean'),
    MedianError=('Абс_ошибка ($)', 'median'),
    MAPE=('Отн_ошибка (%)', 'mean'),
).round(2)
seg_stats.columns = ['Записей', 'MAE ($)', 'Медиана ошибки ($)', 'MAPE (%)']
st.dataframe(seg_stats)

col1, col2 = st.columns(2)
with col1:
    fig_seg = px.bar(seg_stats.reset_index(), x='Сегмент', y='MAE ($)',
                     title='MAE по ценовым сегментам',
                     template='plotly_white', color='MAE ($)',
                     color_continuous_scale='reds',
                     labels={'Сегмент': 'Ценовой сегмент'})
    st.plotly_chart(fig_seg, width='stretch')

with col2:
    fig_mape = px.bar(seg_stats.reset_index(), x='Сегмент', y='MAPE (%)',
                      title='MAPE (%) по ценовым сегментам',
                      template='plotly_white', color='MAPE (%)',
                      color_continuous_scale='oranges')
    st.plotly_chart(fig_mape, width='stretch')

st.info("Наибольшие абсолютные ошибки (MAE) — в сегменте свыше 50 тыс. долл., "
        "но наибольшая относительная ошибка (MAPE) — в сегменте до 500 долл.: "
        "малые выплаты непредсказуемы.")

# ── Паттерны в ошибочных случаях ────────────────────────────────────────────
st.subheader("Паттерны в случаях с наибольшими ошибками")

top_errors = errors_df.nlargest(500, 'Абс_ошибка ($)')
all_rest   = errors_df.nsmallest(len(errors_df)-500, 'Абс_ошибка ($)')

compare_feats = ['InitialCaseEstimate', 'Age_x_WeeklyPay', 'IsFullTime',
                 'HasDependents', 'Log_InitialEstimate']
compare_feats = [f for f in compare_feats if f in top_errors.columns]

if compare_feats:
    compare_df = pd.DataFrame({
        'Признак': compare_feats,
        'Топ-500 ошибок (среднее)': [top_errors[f].mean()
                                      for f in compare_feats],
        'Остальные (среднее)':      [all_rest[f].mean()
                                      for f in compare_feats],
    }).round(4)
    st.dataframe(compare_df)
    st.caption(
        "Сравнение средних значений признаков: "
        "топ-500 самых ошибочных предсказаний vs остальные.")

# ── Интерактивный фильтр ─────────────────────────────────────────────────────
st.subheader("Интерактивный дашборд ошибок")
max_err = float(errors_df['Абс_ошибка ($)'].quantile(0.99))
err_threshold = st.slider("Показать случаи с ошибкой >",
                           0.0, max_err, max_err * 0.5, step=1000.0,
                           format="$%.0f")
filtered = errors_df[errors_df['Абс_ошибка ($)'] > err_threshold]
st.markdown(f"Найдено **{len(filtered)}** случаев с ошибкой > ${err_threshold:,.0f}")

if not filtered.empty:
    fig_f = px.scatter(filtered.reset_index(drop=True),
                       x='Фактическая ($)', y='Предсказанная ($)',
                       color='Отн_ошибка (%)',
                       hover_data=['Абс_ошибка ($)', 'Отн_ошибка (%)'],
                       title=f'Случаи с ошибкой > ${err_threshold:,.0f}',
                       template='plotly_white',
                       color_continuous_scale='reds',
                       log_x=True, log_y=True,
                       opacity=0.6)
    st.plotly_chart(fig_f, width='stretch')
