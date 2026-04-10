"""
Задача 0: Сегментация данных и анализ по группам
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Сегментация", layout="wide")
st.title("🗂️ Сегментация данных и анализ по группам")

with st.spinner("Загрузка данных..."):
    df   = load_data()
    df_p = feature_engineering(preprocess(df))

target = 'UltimateIncurredClaimCost'
X = df_p.drop(columns=[target])
y = df_p[target]

# ── K-Means кластеризация ───────────────────────────────────────────────────
st.subheader("Автоматическая сегментация через K-Means")
n_clusters = st.slider("Количество сегментов:", 2, 6, 3)

scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(X)
kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_p['Segment'] = kmeans.fit_predict(X_scaled)

# Статистика по сегментам
st.subheader("Статистика по сегментам")
seg_stats = df_p.groupby('Segment')[target].agg(
    ['count', 'mean', 'median', 'std']
).round(2)
seg_stats.columns = ['Записей', 'Среднее ($)', 'Медиана ($)', 'Std ($)']
st.dataframe(seg_stats)

# ── Визуализация сегментов ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for seg in range(n_clusters):
    mask = df_p['Segment'] == seg
    vals = df_p.loc[mask, target]
    ax.hist(vals, bins=30, alpha=0.6, label=f'Сегмент {seg}')
ax.set_xlabel("UltimateIncurredClaimCost ($)")
ax.set_ylabel("Количество")
ax.set_title("Распределение выплат по сегментам")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Сравнение: общая модель vs модели по сегментам ─────────────────────────
st.subheader("Общая модель vs Специализированные модели")

if st.button("🚀 Сравнить модели", type="primary"):
    results = {}

    # Общая модель
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, np.log1p(y), test_size=0.2, random_state=42
    )
    gen_model = RandomForestRegressor(n_estimators=50, random_state=42,
                                      n_jobs=-1)
    gen_model.fit(X_tr, y_tr)
    r2_gen = r2_score(y_te, gen_model.predict(X_te))
    results['Общая модель'] = round(r2_gen, 4)

    # Модели по сегментам
    r2_segs = []
    for seg in range(n_clusters):
        mask   = df_p['Segment'] == seg
        X_seg  = X[mask]
        y_seg  = np.log1p(y[mask])
        if len(X_seg) < 50:
            continue
        Xtr, Xte, ytr, yte = train_test_split(
            X_seg, y_seg, test_size=0.2, random_state=42
        )
        m = RandomForestRegressor(n_estimators=50, random_state=42,
                                   n_jobs=-1)
        m.fit(Xtr, ytr)
        r2_segs.append(r2_score(yte, m.predict(Xte)))

    results['Сегментированные (среднее)'] = round(np.mean(r2_segs), 4)

    comp_df = pd.DataFrame.from_dict(
        results, orient='index', columns=['R²']
    )
    st.dataframe(comp_df)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(results.keys()), list(results.values()),
           color=['steelblue', '#4CAF50'])
    ax.set_ylabel("R²")
    ax.set_title("Сравнение подходов")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()