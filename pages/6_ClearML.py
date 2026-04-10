"""
Задача 6: ClearML интеграция — эксперименты, датасет, загрузка модели, предсказания
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib, os
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering

st.set_page_config(page_title="ClearML", layout="wide")
st.title("ClearML — Управление экспериментами и деплой")

st.markdown("""
ClearML отслеживает все ML-эксперименты: метрики, гиперпараметры,
датасеты и модели. Все скрипты запускаются с `clearml-init` — один раз.
""")

with st.spinner("Загрузка данных..."):
    df   = load_data()
    df_p = feature_engineering(preprocess(df))

target = 'UltimateIncurredClaimCost'
X = df_p.drop(columns=[target])
y = np.log1p(df_p[target])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Метрики датасета
col1, col2, col3, col4 = st.columns(4)
col1.metric("Записей",   f"{len(df):,}")
col2.metric("Признаков (после FE)", X.shape[1])
col3.metric("Пропусков", f"{df.isnull().sum().sum():,}")
col4.metric("Целевая переменная", "UltimateIncurredClaimCost")

tab1, tab2, tab3 = st.tabs(["Запуск экспериментов",
                             "Загрузка модели из ClearML",
                             "Предсказание (Inference)"])

# ── TAB 1: Эксперименты ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Запуск и логирование экспериментов")
    st.markdown("""
    Скрипты в `clearml_scripts/` запускают эксперименты с полным логированием:
    метрик, гиперпараметров и графиков в ClearML Web.
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Experiment 1 — Linear Regression baseline",
                     type="primary", key="exp1"):
            with st.spinner("Обучение..."):
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(X_train)
                Xte_s = sc.transform(X_test)
                lr = LinearRegression()
                lr.fit(Xtr_s, y_train)
                yp = lr.predict(Xte_s)
                r2   = r2_score(y_test, yp)
                rmse = float(np.sqrt(mean_squared_error(y_test, yp)))
                mae  = float(mean_absolute_error(
                    np.expm1(y_test), np.expm1(yp)))
            st.success("Experiment 1 завершён!")
            st.session_state['exp1'] = {"R2": round(r2,4),
                                         "RMSE": round(rmse,4),
                                         "MAE_USD": round(mae,0)}
            c1,c2,c3 = st.columns(3)
            c1.metric("R2",   f"{r2:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("MAE",  f"${mae:,.0f}")

            # Попытка залогировать в ClearML
            try:
                from clearml import Task
                task = Task.init(
                    project_name="Workers Compensation",
                    task_name="Experiment 1 — Linear Regression Baseline",
                    task_type=Task.TaskTypes.training,
                    reuse_last_task_id=False)
                task.connect({"model": "LinearRegression", "test_size": 0.2})
                logger = task.get_logger()
                logger.report_scalar("Metrics", "R2",   0, r2)
                logger.report_scalar("Metrics", "RMSE", 0, rmse)
                logger.report_scalar("Metrics", "MAE",  0, mae)
                task.close()
                st.info("Метрики залогированы в ClearML!")
            except Exception as e:
                st.warning(f"ClearML недоступен (локальный режим): {e}")

    with col2:
        if st.button("Experiment 2 — XGBoost (оптимизированный)",
                     type="primary", key="exp2"):
            with st.spinner("Обучение XGBoost..."):
                best_params = {
                    'n_estimators': 200, 'max_depth': 6,
                    'learning_rate': 0.05, 'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42, 'verbosity': 0
                }
                model = xgb.XGBRegressor(**best_params)
                model.fit(X_train, y_train)
                yp   = model.predict(X_test)
                r2   = r2_score(y_test, yp)
                rmse = float(np.sqrt(mean_squared_error(y_test, yp)))
                mae  = float(mean_absolute_error(
                    np.expm1(y_test), np.expm1(yp)))
                joblib.dump(model, "models/XGBoost_optimized.pkl")
            st.success("Experiment 2 завершён!")
            st.session_state['exp2'] = {"R2": round(r2,4),
                                         "RMSE": round(rmse,4),
                                         "MAE_USD": round(mae,0)}
            c1,c2,c3 = st.columns(3)
            c1.metric("R2",   f"{r2:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("MAE",  f"${mae:,.0f}")

            # Feature importance
            fi = pd.Series(model.feature_importances_,
                           index=X.columns).sort_values(ascending=False).head(10)
            fig_fi = px.bar(fi[::-1], orientation='h',
                            title='Топ-10 важных признаков (XGBoost)',
                            labels={'value':'Важность', 'index':'Признак'},
                            template='plotly_white')
            st.plotly_chart(fig_fi, use_container_width=True)

            try:
                from clearml import Task, OutputModel
                task = Task.init(
                    project_name="Workers Compensation",
                    task_name="Experiment 2 — XGBoost Optimized",
                    task_type=Task.TaskTypes.training,
                    reuse_last_task_id=False)
                task.connect(best_params)
                logger = task.get_logger()
                logger.report_scalar("Metrics", "R2",   0, r2)
                logger.report_scalar("Metrics", "RMSE", 0, rmse)
                logger.report_scalar("Metrics", "MAE",  0, mae)
                out_model = OutputModel(task=task, framework="XGBoost")
                out_model.update_weights("models/XGBoost_optimized.pkl")
                task.close()
                st.info("Метрики и модель залогированы в ClearML!")
            except Exception as e:
                st.warning(f"ClearML недоступен (локальный режим): {e}")

    if 'exp1' in st.session_state or 'exp2' in st.session_state:
        st.subheader("Сводная таблица экспериментов")
        rows = []
        if 'exp1' in st.session_state:
            rows.append({"Эксперимент": "Experiment 1 — Linear Regression",
                         **st.session_state['exp1'], "Статус": "Завершён"})
        if 'exp2' in st.session_state:
            rows.append({"Эксперимент": "Experiment 2 — XGBoost",
                         **st.session_state['exp2'], "Статус": "Завершён"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Запуск через командную строку (ClearML Web)")
    st.code("""
# 1. Создать датасет
python clearml_scripts/dataset_creation.py

# 2. Эксперимент 1 (Linear Regression baseline)
python clearml_scripts/experiment1.py

# 3. Эксперимент 2 (XGBoost + Optuna)
python clearml_scripts/experiment2.py
    """, language="bash")
    st.markdown("[Открыть ClearML Web](https://app.clear.ml)")

# ── TAB 2: Загрузка модели из ClearML ────────────────────────────────────────
with tab2:
    st.subheader("Загрузка обученной модели из ClearML по ID")
    st.markdown("""
    После запуска экспериментов в ClearML появляются зарегистрированные модели.
    Скопируйте **Model ID** из ClearML Web → Models и вставьте ниже.
    """)

    model_id = st.text_input(
        "ID модели из ClearML (например: abc123def456)",
        placeholder="Введите Model ID из https://app.clear.ml")

    if st.button("Загрузить модель из ClearML", key="load_model"):
        if not model_id.strip():
            st.error("Введите Model ID!")
        else:
            try:
                from clearml import Model
                with st.spinner("Загрузка модели из ClearML..."):
                    cm = Model(model_id=model_id.strip())
                    model_path = cm.get_local_copy()
                    loaded = joblib.load(model_path)
                    st.session_state['clearml_model'] = loaded
                    st.session_state['clearml_model_name'] = cm.name
                st.success(f"Модель '{cm.name}' загружена!")
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")
                st.info("Убедитесь, что Model ID правильный и модель "
                        "зарегистрирована в ClearML.")

    # Альтернатива: загрузка локального .pkl
    st.markdown("**Или** загрузите локальный .pkl файл:")
    pkl_files = [f for f in os.listdir("models") if f.endswith(".pkl")] \
        if os.path.exists("models") else []
    if pkl_files:
        selected_pkl = st.selectbox("Выберите локальный файл модели:", pkl_files)
        if st.button("Загрузить локальную модель", key="load_local"):
            loaded = joblib.load(f"models/{selected_pkl}")
            st.session_state['clearml_model'] = loaded
            st.session_state['clearml_model_name'] = selected_pkl
            st.success(f"Модель '{selected_pkl}' загружена!")

    if 'clearml_model' in st.session_state:
        st.info(f"Активная модель: **{st.session_state.get('clearml_model_name', '?')}**")
        # Быстрая проверка
        try:
            y_pred_check = st.session_state['clearml_model'].predict(X_test)
            r2_check = r2_score(y_test, y_pred_check)
            st.metric("R2 загруженной модели на тестовой выборке",
                      f"{r2_check:.4f}")
        except Exception as e:
            st.warning(f"Не удалось оценить модель: {e}")

# ── TAB 3: Prediction UI ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Предсказание стоимости страховой выплаты")

    if 'clearml_model' not in st.session_state:
        st.warning("Сначала загрузите модель на вкладке 'Загрузка модели'.")
    else:
        st.success(f"Используется модель: "
                   f"{st.session_state.get('clearml_model_name','?')}")
        st.markdown("Введите данные страхового случая:")

        col1, col2, col3 = st.columns(3)
        age             = col1.number_input("Возраст (лет)", 18, 80, 35)
        weekly_pay      = col2.number_input("Недельная зарплата ($)", 100, 5000, 500)
        initial_est     = col3.number_input("Начальная оценка дела ($)", 0, 500000, 5000)

        col4, col5, col6 = st.columns(3)
        hours_week      = col4.number_input("Часов в неделю", 1, 80, 40)
        days_week       = col5.number_input("Дней в неделю", 1, 7, 5)
        gender          = col6.selectbox("Пол", ["M", "F"])

        col7, col8, col9 = st.columns(3)
        marital         = col7.selectbox("Семейное положение",
                                         ["M", "S", "D", "W"])
        dep_children    = col8.number_input("Детей-иждивенцев", 0, 10, 0)
        dep_other       = col9.number_input("Других иждивенцев", 0, 10, 0)

        col10, col11 = st.columns(2)
        acc_year  = col10.number_input("Год несчастного случая", 1990, 2024, 2010)
        acc_month = col11.slider("Месяц несчастного случая", 1, 12, 6)

        if st.button("Рассчитать прогноз", type="primary", key="predict"):
            from sklearn.preprocessing import LabelEncoder

            # Сборка вектора признаков в том же порядке, что и при обучении
            input_dict = {
                'Age': age,
                'Gender': gender,
                'MaritalStatus': marital,
                'DependentChildren': dep_children,
                'DependentsOther': dep_other,
                'WeeklyPay': weekly_pay,
                'PartTimeFullTime': 'F' if hours_week >= 35 else 'P',
                'HoursWorkedPerWeek': hours_week,
                'DaysWorkedPerWeek': days_week,
                'InitialCaseEstimate': initial_est,
                'Accident_Year': acc_year,
                'Accident_Month': acc_month,
                'Accident_DayOfWeek': 0,
                'Reported_Year': acc_year,
                'Reported_Month': acc_month,
                'Reported_DayOfWeek': 0,
                'ReportDelay_Days': 30,
            }
            input_df = pd.DataFrame([input_dict])

            # Кодирование категорий
            cat_cols = ['Gender', 'MaritalStatus', 'PartTimeFullTime']
            for c in cat_cols:
                if c in input_df.columns:
                    input_df[c] = LabelEncoder().fit_transform(
                        input_df[c].astype(str))

            # Feature engineering
            input_df['Age_x_WeeklyPay']  = input_df['Age'] * input_df['WeeklyPay']
            input_df['Estimate_per_Pay'] = (input_df['InitialCaseEstimate'] /
                                            (input_df['WeeklyPay'] + 1))
            input_df['HasDependents']    = int(dep_other > 0 or dep_children > 0)
            input_df['IsFullTime']       = int(hours_week >= 35)
            input_df['Log_InitialEstimate'] = np.log1p(initial_est)

            # Выравниваем столбцы по обучающей выборке
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X.columns]

            try:
                pred_log = st.session_state['clearml_model'].predict(input_df)
                pred_usd = float(np.expm1(pred_log[0]))
                st.metric("Прогнозируемая стоимость выплаты",
                          f"${pred_usd:,.0f}")
                if pred_usd < 1000:
                    st.success("Низкозатратный случай")
                elif pred_usd < 20000:
                    st.warning("Среднезатратный случай")
                else:
                    st.error("Высокозатратный случай — требует особого внимания!")
            except Exception as e:
                st.error(f"Ошибка предсказания: {e}")
