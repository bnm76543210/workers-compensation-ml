"""
Задача 1: Детальный анализ датасета
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data

st.set_page_config(page_title="Анализ данных", layout="wide")
st.title("📊 Детальный анализ датасета Workers Compensation")

with st.spinner("Загрузка данных..."):
    df = load_data()

# ── Общая статистика ────────────────────────────────────────────────────────
st.subheader("Общая информация")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Записей", f"{len(df):,}")
c2.metric("Признаков", df.shape[1])
c3.metric("Пропусков", f"{df.isnull().sum().sum():,}")
c4.metric("Средняя выплата",
          f"${df['UltimateIncurredClaimCost'].mean():,.0f}")

st.subheader("Первые 5 строк")
st.dataframe(df.head())

st.subheader("Описательная статистика")
st.dataframe(df.describe().round(2))

# ── Распределение целевой переменной ───────────────────────────────────────
st.subheader("Распределение целевой переменной (UltimateIncurredClaimCost)")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
y = df['UltimateIncurredClaimCost'].dropna()
axes[0].hist(y, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title("Оригинальное распределение")
axes[0].set_xlabel("Стоимость ($)")
axes[1].hist(np.log1p(y), bins=50, color='#4CAF50', edgecolor='white')
axes[1].set_title("Log-преобразование")
axes[1].set_xlabel("log(1 + Стоимость)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Корреляционный анализ ───────────────────────────────────────────────────
st.subheader("Корреляционный анализ")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Матрица корреляций")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Анализ категориальных признаков ────────────────────────────────────────
st.subheader("Анализ по категориальным признакам")
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'ClaimDescription' in cat_cols:
    cat_cols.remove('ClaimDescription')

if cat_cols:
    selected_cat = st.selectbox("Выберите признак:", cat_cols)
    fig, ax = plt.subplots(figsize=(12, 5))
    df.boxplot(column='UltimateIncurredClaimCost',
               by=selected_cat, ax=ax)
    ax.set_title(f"Стоимость выплат по {selected_cat}")
    ax.set_xlabel(selected_cat)
    ax.set_ylabel("Стоимость ($)")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Scatter: InitialEstimate vs Cost ───────────────────────────────────────
if 'InitialCaseEstimate' in df.columns:
    st.subheader("Начальная оценка vs Итоговая стоимость")
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(min(2000, len(df)), random_state=42)
    ax.scatter(sample['InitialCaseEstimate'],
               sample['UltimateIncurredClaimCost'],
               alpha=0.3, color='steelblue', s=15)
    ax.set_xlabel("Начальная оценка ($)")
    ax.set_ylabel("Итоговая стоимость ($)")
    ax.set_title("Точность начальных оценок страховых дел")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()