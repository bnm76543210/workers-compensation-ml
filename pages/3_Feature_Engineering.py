"""
Задача 3: Feature Engineering — создание новых признаков
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering

st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("🔧 Feature Engineering")
st.markdown("Создание дополнительных признаков для улучшения качества модели.")

with st.spinner("Загрузка данных..."):
    df     = load_data()
    df_raw = preprocess(df)
    df_fe  = feature_engineering(df_raw.copy())

target = 'UltimateIncurredClaimCost'

# ── Сравнение до/после ──────────────────────────────────────────────────────
st.subheader("Количество признаков до и после Feature Engineering")
c1, c2, c3 = st.columns(3)
c1.metric("До обработки",   df_raw.shape[1] - 1)
c2.metric("После FE",       df_fe.shape[1] - 1)
c3.metric("Новых признаков", df_fe.shape[1] - df_raw.shape[1])

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
st.dataframe(desc_df, use_container_width=True)

# ── Визуализация новых признаков ────────────────────────────────────────────
st.subheader("Визуализация новых признаков")

available = [f for f in new_features if f in df_fe.columns]
if available:
    selected = st.selectbox("Выберите признак для анализа:", available)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Распределение признака
    axes[0].hist(df_fe[selected].dropna(), bins=40,
                 color='steelblue', edgecolor='white')
    axes[0].set_title(f"Распределение: {selected}")
    axes[0].set_xlabel(selected)
    axes[0].set_ylabel("Количество")

    # Scatter с целевой переменной
    sample = df_fe[[selected, target]].dropna().sample(
        min(2000, len(df_fe)), random_state=42
    )
    axes[1].scatter(sample[selected], sample[target],
                    alpha=0.3, color='steelblue', s=10)
    axes[1].set_xlabel(selected)
    axes[1].set_ylabel("UltimateIncurredClaimCost ($)")
    axes[1].set_title(f"{selected} vs Стоимость выплат")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Корреляция новых признаков с целевой переменной ────────────────────────
st.subheader("Корреляция новых признаков с целевой переменной")
num_df = df_fe.select_dtypes(include=[np.number])
corr_target = num_df.corr()[target].drop(target).sort_values(
    key=abs, ascending=False
).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#4CAF50' if v > 0 else '#F44336' for v in corr_target.values]
ax.barh(corr_target.index[::-1], corr_target.values[::-1], color=colors[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Корреляция Пирсона")
ax.set_title("Топ-15 признаков по корреляции с целевой переменной")
plt.tight_layout()
st.pyplot(fig)
plt.close()