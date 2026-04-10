"""
Задача 2: Детальный анализ датасета
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from src.data_loader import load_data

st.set_page_config(page_title="Анализ данных", layout="wide")
st.title("📊 Детальный анализ датасета Workers Compensation")

with st.spinner("Загрузка данных..."):
    df = load_data()

target = 'UltimateIncurredClaimCost'

# ── Общая статистика ────────────────────────────────────────────────────────
st.subheader("Общая информация")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Записей", f"{len(df):,}")
c2.metric("Признаков", df.shape[1])
c3.metric("Пропусков", f"{df.isnull().sum().sum():,}")
c4.metric("Средняя выплата", f"${df[target].mean():,.0f}")

st.subheader("Первые 5 строк")
st.dataframe(df.head())

st.subheader("Описательная статистика")
st.dataframe(df.describe().round(2))

# ── Распределение целевой переменной ────────────────────────────────────────
st.subheader("Распределение целевой переменной")
y = df[target].dropna()

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Histogram(x=y, nbinsx=80,
                                 marker_color='steelblue', opacity=0.75))
    fig.update_layout(title="Оригинальное распределение выплат (USD)",
                      xaxis_title="Стоимость ($)", yaxis_title="Количество",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure(go.Histogram(x=np.log1p(y), nbinsx=60,
                                  marker_color='#4CAF50', opacity=0.75))
    fig2.update_layout(title="Log-преобразование log(1+y)",
                       xaxis_title="log(1 + Стоимость)",
                       template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# Нормальность и выбросы
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Тест на нормальность (D'Agostino-Pearson)")
    log_y = np.log1p(y)
    stat, p_value = stats.normaltest(
        log_y.sample(min(5000, len(log_y)), random_state=42))
    st.markdown(f"Log-переменная: statistic=`{stat:.4f}`, p-value=`{p_value:.6f}`")
    if p_value < 0.05:
        st.warning("Распределение **не является** нормальным (p < 0.05)")
    else:
        st.success("Распределение близко к нормальному (p >= 0.05)")

with col2:
    st.markdown("#### Выбросы по методу IQR")
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)).sum()
    st.markdown(f"Q1=`${Q1:,.0f}` | Q3=`${Q3:,.0f}` | IQR=`${IQR:,.0f}`")
    st.metric("Количество выбросов", f"{n_out:,} ({n_out/len(y)*100:.1f}%)")

# ── Временной анализ ─────────────────────────────────────────────────────────
st.subheader("Временной анализ — тренды по годам и месяцам")
acc_dt = pd.to_datetime(df['DateTimeOfAccident'], errors='coerce')
df_time = df.copy()
df_time['AccidentYear']  = acc_dt.dt.year
df_time['AccidentMonth'] = acc_dt.dt.month

col1, col2 = st.columns(2)
with col1:
    yearly = (df_time[df_time['AccidentYear'].between(1985, 2015)]
              .groupby('AccidentYear')[target]
              .agg(['mean', 'median']).reset_index())
    yearly.columns = ['Год', 'Среднее', 'Медиана']
    fig_y = px.line(yearly, x='Год', y=['Среднее', 'Медиана'],
                    title='Средняя и медианная выплата по годам',
                    labels={'value': 'Стоимость ($)', 'variable': 'Метрика'},
                    template='plotly_white')
    st.plotly_chart(fig_y, use_container_width=True)

with col2:
    MONTHS = ['Янв','Фев','Мар','Апр','Май','Июн',
              'Июл','Авг','Сен','Окт','Ноя','Дек']
    monthly = (df_time.groupby('AccidentMonth')[target]
               .mean().reset_index())
    monthly.columns = ['Месяц', 'Среднее']
    monthly['МесяцИмя'] = monthly['Месяц'].apply(
        lambda x: MONTHS[int(x)-1] if 1 <= x <= 12 else str(x))
    fig_m = px.bar(monthly, x='МесяцИмя', y='Среднее',
                   title='Средняя выплата по месяцам (сезонность)',
                   labels={'Среднее': 'Средняя выплата ($)'},
                   template='plotly_white', color='Среднее',
                   color_continuous_scale='blues')
    st.plotly_chart(fig_m, use_container_width=True)

cnt = (df_time[df_time['AccidentYear'].between(1990, 2015)]
       .groupby('AccidentYear').size().reset_index(name='Количество'))
fig_cnt = px.bar(cnt, x='AccidentYear', y='Количество',
                 title='Количество страховых дел по годам',
                 template='plotly_white', color='Количество',
                 color_continuous_scale='viridis',
                 labels={'AccidentYear': 'Год'})
st.plotly_chart(fig_cnt, use_container_width=True)

# ── Корреляционный анализ ────────────────────────────────────────────────────
st.subheader("Корреляционный анализ")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Матрица корреляций")
plt.tight_layout()
st.pyplot(fig)
plt.close()

corr_target = corr[target].drop(target).sort_values(key=abs, ascending=False)
fig_bar = px.bar(x=corr_target.values, y=corr_target.index,
                 orientation='h',
                 title='Корреляция признаков с целевой переменной',
                 labels={'x': 'Корреляция Пирсона', 'y': 'Признак'},
                 color=corr_target.values, color_continuous_scale='RdBu_r',
                 template='plotly_white')
fig_bar.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Анализ категориальных признаков ─────────────────────────────────────────
st.subheader("Анализ по категориальным признакам")
cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns
            if c != 'ClaimDescription']
if cat_cols:
    selected_cat = st.selectbox("Выберите признак:", cat_cols)
    fig_box = px.box(df, x=selected_cat, y=target,
                     title=f"Стоимость выплат по {selected_cat}",
                     template='plotly_white',
                     labels={target: 'Стоимость ($)'}, log_y=True)
    st.plotly_chart(fig_box, use_container_width=True)

    grp = df.groupby(selected_cat)[target].agg(
        ['mean','median','count']).round(2)
    grp.columns = ['Среднее ($)', 'Медиана ($)', 'Количество дел']
    st.dataframe(grp.sort_values('Среднее ($)', ascending=False))

# ── Scatter: InitialEstimate vs Cost ─────────────────────────────────────────
if 'InitialCaseEstimate' in df.columns:
    st.subheader("Начальная оценка vs Итоговая стоимость")
    sample = df.sample(min(3000, len(df)), random_state=42)
    accuracy = (sample['InitialCaseEstimate'] /
                sample[target].replace(0, np.nan)).dropna()
    pct_2x = ((accuracy >= 0.5) & (accuracy <= 2.0)).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Корреляция Pearson",
              f"{df['InitialCaseEstimate'].corr(df[target]):.3f}")
    c2.metric("В пределах 2x от факта", f"{pct_2x:.1f}%")
    c3.metric("Медиана отношения оценка/факт", f"{accuracy.median():.2f}x")

    max_val = float(max(sample['InitialCaseEstimate'].max(),
                        sample[target].max()))
    fig_sc = px.scatter(sample, x='InitialCaseEstimate', y=target,
                        opacity=0.3, title='Точность начальных оценок',
                        labels={'InitialCaseEstimate': 'Начальная оценка ($)',
                                target: 'Итоговая стоимость ($)'},
                        template='plotly_white', log_x=True, log_y=True)
    fig_sc.add_trace(go.Scatter(x=[1, max_val], y=[1, max_val], mode='lines',
                                name='Идеальная оценка',
                                line=dict(color='red', dash='dash')))
    st.plotly_chart(fig_sc, use_container_width=True)
