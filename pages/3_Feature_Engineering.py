"""
Задача 3: Feature Engineering — предобработка и создание признаков
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
from src.logger import log
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("Предобработка и Feature Engineering")
log("=== Страница 3: Feature Engineering загружена ===")

with st.spinner("Загрузка данных..."):
    df     = load_data()
    df_raw = preprocess(df)
    df_fe  = feature_engineering(df_raw.copy())

target = 'UltimateIncurredClaimCost'

# ── Сравнение до/после ──────────────────────────────────────────────────────
st.subheader("Количество признаков до и после Feature Engineering")
c1, c2, c3 = st.columns(3)
log("Feature Engineering статистика",
    before=df.shape[1]-1, after=df_fe.shape[1]-1)
c1.metric("До обработки (исходных)", df.shape[1] - 1)
c2.metric("После FE", df_fe.shape[1] - 1)
c3.metric("Новых признаков создано", df_fe.shape[1] - df_raw.shape[1])

# ── Список новых признаков ──────────────────────────────────────────────────
new_features = [c for c in df_fe.columns if c not in df_raw.columns]
st.subheader("Созданные признаки")
fe_desc = {
    'Age_x_WeeklyPay':      'Взаимодействие возраста и зарплаты',
    'Estimate_per_Pay':     'Отношение начальной оценки к зарплате',
    'HasDependents':        'Бинарный: есть иждивенцы (1/0)',
    'IsFullTime':           'Бинарный: полная занятость (>=35 ч/нед)',
    'Log_InitialEstimate':  'Лог-преобразование начальной оценки',
    'Accident_Year':        'Год несчастного случая',
    'Accident_Month':       'Месяц несчастного случая',
    'Accident_DayOfWeek':   'День недели несчастного случая',
    'Reported_Year':        'Год подачи заявки',
    'Reported_Month':       'Месяц подачи заявки',
    'Reported_DayOfWeek':   'День недели подачи заявки',
    'ReportDelay_Days':     'Задержка подачи заявки (дней)',
}
desc_df = pd.DataFrame([
    {"Признак": f, "Описание": fe_desc.get(f, "—")}
    for f in new_features
])
st.dataframe(desc_df, width='stretch')

# ── Обработка выбросов (IQR) ─────────────────────────────────────────────────
st.subheader("Обработка выбросов (IQR-метод)")
y = df[target].dropna()
Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
mask_out = (y < lower) | (y > upper)
n_out = mask_out.sum()

col1, col2, col3 = st.columns(3)
col1.metric("Выбросов (IQR)", f"{n_out:,} ({n_out/len(y)*100:.1f}%)")
col2.metric("Нижняя граница", f"${lower:,.0f}")
col3.metric("Верхняя граница", f"${upper:,.0f}")

fig_box = go.Figure()
fig_box.add_trace(go.Box(y=y, name="Все данные",
                          marker_color='steelblue', boxpoints='outliers'))
fig_box.update_layout(title="Boxplot целевой переменной (выбросы выделены)",
                      yaxis_title="Стоимость ($)", template="plotly_white")
st.plotly_chart(fig_box, width='stretch')

st.info(f"В модели применяется **log1p-преобразование** целевой переменной, "
        f"что нивелирует влияние выбросов без их удаления.")

# ── Визуализация новых признаков ────────────────────────────────────────────
st.subheader("Визуализация созданных признаков")
available = [f for f in new_features if f in df_fe.columns]
if available:
    selected = st.selectbox("Выберите признак для анализа:", available)
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(df_fe, x=selected, nbins=40,
                                title=f"Распределение: {selected}",
                                template='plotly_white',
                                color_discrete_sequence=['steelblue'])
        st.plotly_chart(fig_hist, width='stretch')
    with col2:
        sample = df_fe[[selected, target]].dropna().sample(
            min(2000, len(df_fe)), random_state=42)
        fig_sc = px.scatter(sample, x=selected, y=target,
                            opacity=0.3,
                            title=f"{selected} vs Стоимость выплат",
                            labels={target: 'UltimateIncurredClaimCost ($)'},
                            template='plotly_white',
                            trendline='ols')
        st.plotly_chart(fig_sc, width='stretch')

# ── Корреляция признаков с целевой ──────────────────────────────────────────
st.subheader("Корреляция признаков с целевой переменной")
num_df = df_fe.select_dtypes(include=[np.number])
corr_target = (num_df.corr()[target].drop(target)
               .sort_values(key=abs, ascending=False).head(15))
fig_c = px.bar(x=corr_target.values, y=corr_target.index,
               orientation='h',
               title='Топ-15 признаков по корреляции с целевой переменной',
               labels={'x': 'Корреляция Пирсона', 'y': 'Признак'},
               color=corr_target.values,
               color_continuous_scale='RdBu_r',
               template='plotly_white')
fig_c.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig_c, width='stretch')

# ── PCA ─────────────────────────────────────────────────────────────────────
st.subheader("PCA — Уменьшение размерности")
st.markdown("Проекция данных в 2D через Principal Component Analysis.")

X_pca = df_fe.drop(columns=[target]).select_dtypes(include=[np.number])
X_pca = X_pca.fillna(X_pca.median())

n_pca = st.slider("Количество компонент PCA:", 2, min(10, X_pca.shape[1]), 5)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=n_pca)
components = pca.fit_transform(X_scaled)

# Объяснённая дисперсия
fig_var = px.bar(x=[f"PC{i+1}" for i in range(n_pca)],
                 y=pca.explained_variance_ratio_ * 100,
                 title="Объяснённая дисперсия по компонентам PCA",
                 labels={'x': 'Компонента', 'y': 'Объяснённая дисперсия (%)'},
                 template='plotly_white',
                 color=pca.explained_variance_ratio_,
                 color_continuous_scale='blues')
st.plotly_chart(fig_var, width='stretch')
st.info(f"Две первые компоненты объясняют "
        f"{pca.explained_variance_ratio_[:2].sum()*100:.1f}% дисперсии.")

# Scatter PCA-1 vs PCA-2
sample_idx = np.random.RandomState(42).choice(len(components),
                                               min(3000, len(components)),
                                               replace=False)
y_log = np.log1p(df_fe[target].values)
fig_pca = px.scatter(x=components[sample_idx, 0],
                     y=components[sample_idx, 1],
                     color=y_log[sample_idx],
                     opacity=0.5,
                     title="PCA: PC1 vs PC2 (цвет = log(стоимость))",
                     labels={'x': 'PC1', 'y': 'PC2', 'color': 'log(стоимость)'},
                     color_continuous_scale='viridis',
                     template='plotly_white')
st.plotly_chart(fig_pca, width='stretch')

# ── t-SNE ───────────────────────────────────────────────────────────────────
st.subheader("t-SNE — Нелинейная визуализация многомерных данных")
st.markdown("t-SNE проецирует 22 признака в 2D, сохраняя локальную структуру данных.")
st.warning("t-SNE вычисляется на 1000 наблюдениях — это может занять ~30 секунд.")

if st.button("Запустить t-SNE", type="primary", key="btn_tsne"):
    log("Пользователь: нажата кнопка t-SNE")
    with st.spinner("Вычисление t-SNE (1000 наблюдений)..."):
        idx_tsne = np.random.RandomState(42).choice(
            len(X_scaled), min(1000, len(X_scaled)), replace=False)
        # PCA до 30 компонент перед t-SNE (ускорение)
        pre_pca = PCA(n_components=min(30, X_scaled.shape[1]),
                      random_state=42)
        X_pre = pre_pca.fit_transform(X_scaled[idx_tsne])
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=30, n_iter=500)
        emb = tsne.fit_transform(X_pre)

    y_color = np.log1p(df_fe[target].values[idx_tsne])
    fig_tsne = px.scatter(x=emb[:, 0], y=emb[:, 1],
                          color=y_color, opacity=0.6,
                          title="t-SNE визуализация (цвет = log(стоимость))",
                          labels={'x': 't-SNE 1', 'y': 't-SNE 2',
                                  'color': 'log(стоимость)'},
                          color_continuous_scale='plasma',
                          template='plotly_white')
    st.plotly_chart(fig_tsne, width='stretch')
    st.success("t-SNE построен! Видимые кластеры соответствуют группам "
               "страховых случаев с похожими характеристиками.")
