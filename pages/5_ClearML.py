"""
Задача 5: ClearML интеграция — просмотр экспериментов
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ClearML", layout="wide")
st.title("📡 ClearML — Управление экспериментами")
st.markdown("""
ClearML используется для отслеживания всех экспериментов проекта,
версионирования датасетов и логирования метрик.
""")

st.subheader("🗂️ Датасет в ClearML")
st.markdown("""
Датасет **Workers Compensation Dataset** создан и загружен в ClearML:
- **Проект:** Workers Compensation
- **Записей:** ~45 000
- **Признаков:** 16
- **Целевая переменная:** UltimateIncurredClaimCost
""")

st.info("📸 Скриншоты экспериментов из веб-интерфейса ClearML "
        "доступны в разделе отчёта.")

st.subheader("📊 Результаты экспериментов")
experiments = pd.DataFrame({
    "Эксперимент":  ["Experiment 1 — Linear Regression",
                     "Experiment 2 — XGBoost (Optuna)"],
    "R²":           [None, None],
    "RMSE":         [None, None],
    "MAE":          [None, None],
    "Статус":       ["✅ Завершён", "✅ Завершён"],
})
st.dataframe(experiments, use_container_width=True)
st.caption("Метрики обновятся после запуска clearml_scripts/")

st.subheader("🔗 Ссылки")
st.markdown("""
- **ClearML Web:** https://app.clear.ml
- **Проект:** Workers Compensation
- **Запуск скриптов:**
```bash
python clearml_scripts/dataset_creation.py
python clearml_scripts/experiment1.py
python clearml_scripts/experiment2.py
```
""")