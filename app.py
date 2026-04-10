"""
Главная страница Streamlit-приложения
Workers Compensation ML System
"""
import streamlit as st

st.set_page_config(
    page_title="Workers Compensation ML",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Система прогнозирования страховых выплат")
st.markdown("""
Продвинутая ML-система для прогнозирования итоговой стоимости страхового
возмещения **(UltimateIncurredClaimCost)** на основе датасета
**Workers Compensation** (OpenML ID: 42876).
""")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.info("🗂️ **Задача 1**\nСегментация данных")
col2.info("📊 **Задача 2**\nДетальный анализ данных")
col3.info("🔧 **Задача 3**\nFeature Engineering")
col4.info("🤖 **Задача 4**\nМощные модели (XGBoost, LightGBM)")

col5, col6, col7, col8 = st.columns(4)
col5.info("⚙️ **Задача 5**\nOptuna оптимизация")
col6.info("📡 **Задача 6**\nClearML интеграция")
col7.info("💡 **Задача 7**\nSHAP интерпретируемость")
col8.info("🔍 **Задача 8**\nАнализ ошибок")

st.markdown("---")
st.markdown("### Навигация")
st.markdown("""
Используй боковое меню слева для перехода между разделами:
- **Анализ данных** — статистика, распределения, корреляции
- **Feature Engineering** — созданные признаки
- **Модели** — обучение и сравнение моделей
- **Оптимизация** — Optuna подбор гиперпараметров
- **ClearML** — управление экспериментами
- **SHAP** — интерпретируемость предсказаний
- **Анализ ошибок** — детальный разбор ошибок модели
""")
st.caption("Датасет: Workers Compensation | OpenML ID: 42876 | "
           "Выполнил: Фахрутдинов Марат Альбертович")