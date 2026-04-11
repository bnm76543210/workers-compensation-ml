"""
Задача 5: Оптимизация гиперпараметров — Optuna + K-Fold CV + Early Stopping
"""
import streamlit as st
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
from src.logger import log
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(page_title="Оптимизация", layout="wide")
st.title("Оптимизация гиперпараметров — Optuna + K-Fold CV")
log("=== Страница 5: Оптимизация загружена ===")

with st.spinner("Загрузка данных..."):
    df = load_data()
df_proc = feature_engineering(preprocess(df))
target = 'UltimateIncurredClaimCost'
X = df_proc.drop(columns=[target])
y = np.log1p(df_proc[target])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

tab1, tab2, tab3 = st.tabs(["K-Fold кросс-валидация",
                             "Optuna оптимизация",
                             "Early Stopping (LightGBM)"])

# ── TAB 1: K-Fold CV ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("K-Fold Cross-Validation")
    st.markdown("Оценка стабильности модели на нескольких разбиениях данных.")

    col1, col2 = st.columns(2)
    k_folds = col1.slider("Количество фолдов (K):", 3, 10, 5)
    cv_model_name = col2.selectbox(
        "Модель для кросс-валидации:",
        ["XGBoost", "LightGBM"])

    if st.button("Запустить K-Fold CV", type="primary", key="kfold"):
        log("Пользователь: нажата кнопка K-Fold CV")
        if cv_model_name == "XGBoost":
            cv_model = xgb.XGBRegressor(n_estimators=100, random_state=42,
                                         verbosity=0)
        else:
            cv_model = lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                          verbose=-1)

        with st.spinner(f"Запуск {k_folds}-Fold CV..."):
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_scores = []
            fold_rmse   = []
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                cv_model.fit(Xtr, ytr)
                yp   = cv_model.predict(Xval)
                r2   = r2_score(yval, yp)
                rmse = float(np.sqrt(mean_squared_error(yval, yp)))
                fold_scores.append(r2)
                fold_rmse.append(rmse)

        fold_df = pd.DataFrame({
            'Фолд':    [f"Fold {i+1}" for i in range(k_folds)],
            'R2':      [round(s, 4) for s in fold_scores],
            'RMSE':    [round(r, 4) for r in fold_rmse],
        })

        c1, c2, c3 = st.columns(3)
        c1.metric("Среднее R2",   f"{np.mean(fold_scores):.4f}")
        c2.metric("Std R2",       f"±{np.std(fold_scores):.4f}")
        c3.metric("Среднее RMSE", f"{np.mean(fold_rmse):.4f}")

        st.dataframe(fold_df)

        fig_kf = px.bar(fold_df, x='Фолд', y='R2',
                        title=f"{k_folds}-Fold CV: R2 по фолдам",
                        template='plotly_white', color='R2',
                        color_continuous_scale='greens')
        fig_kf.add_hline(y=np.mean(fold_scores), line_dash='dash',
                         line_color='red',
                         annotation_text=f"Среднее R2={np.mean(fold_scores):.4f}")
        st.plotly_chart(fig_kf, width='stretch')

        st.info(f"Малое стандартное отклонение ({np.std(fold_scores):.4f}) "
                "говорит о стабильности модели — она не переобучается.")

# ── TAB 2: Optuna ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Optuna — байесовская оптимизация гиперпараметров")

    col1, col2 = st.columns(2)
    n_trials = col1.slider("Количество итераций:", 10, 100, 30, 10)
    opt_model = col2.selectbox("Модель для оптимизации:",
                               ["XGBoost", "LightGBM"])

    if st.button("Запустить оптимизацию", type="primary", key="optuna"):
        log("Пользователь: нажата кнопка Optuna")
        def objective_xgb(trial):
            params = {
                'n_estimators':     trial.suggest_int('n_estimators', 50, 300),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate',
                                                         0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                         0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42, 'verbosity': 0
            }
            scores = cross_val_score(xgb.XGBRegressor(**params),
                                     X_train, y_train, cv=3, scoring='r2')
            return scores.mean()

        def objective_lgb(trial):
            params = {
                'n_estimators':   trial.suggest_int('n_estimators', 50, 300),
                'max_depth':      trial.suggest_int('max_depth', 3, 12),
                'learning_rate':  trial.suggest_float('learning_rate',
                                                       0.01, 0.3, log=True),
                'num_leaves':     trial.suggest_int('num_leaves', 20, 150),
                'subsample':      trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                         0.6, 1.0),
                'random_state': 42, 'verbose': -1
            }
            scores = cross_val_score(lgb.LGBMRegressor(**params),
                                     X_train, y_train, cv=3, scoring='r2')
            return scores.mean()

        objective = objective_xgb if opt_model == "XGBoost" else objective_lgb

        with st.spinner(f"Optuna: {n_trials} итераций для {opt_model}..."):
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        log("Optuna завершена", best_R2=round(study.best_value, 4), **best)
        st.success(f"Оптимизация завершена! Лучший R2 (CV): "
                   f"{study.best_value:.4f}")

        st.subheader("Лучшие гиперпараметры")
        st.json(best)

        # Обучаем с лучшими параметрами
        if opt_model == "XGBoost":
            best_model = xgb.XGBRegressor(**best, random_state=42, verbosity=0)
            base_model  = xgb.XGBRegressor(n_estimators=100, random_state=42,
                                            verbosity=0)
        else:
            best_model = lgb.LGBMRegressor(**best, random_state=42, verbose=-1)
            base_model  = lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                             verbose=-1)

        best_model.fit(X_train, y_train)
        base_model.fit(X_train, y_train)

        r2_opt  = r2_score(y_test, best_model.predict(X_test))
        r2_base = r2_score(y_test, base_model.predict(X_test))
        rmse_opt  = float(np.sqrt(mean_squared_error(y_test,
                                   best_model.predict(X_test))))
        rmse_base = float(np.sqrt(mean_squared_error(y_test,
                                   base_model.predict(X_test))))

        comp = pd.DataFrame({
            "Модель":  [f"{opt_model} (базовый)", f"{opt_model} (Optuna)"],
            "R2":      [round(r2_base, 4), round(r2_opt, 4)],
            "RMSE":    [round(rmse_base, 4), round(rmse_opt, 4)],
            "Улучшение R2": ["—", f"+{(r2_opt-r2_base):.4f}"],
        })
        st.subheader("До и после оптимизации")
        st.dataframe(comp)

        # История
        vals = [t.value for t in study.trials]
        best_so_far = [max(vals[:i+1]) for i in range(len(vals))]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(y=vals, mode='markers', name='R2 попытки',
                                      marker=dict(color='steelblue', size=5)))
        fig_hist.add_trace(go.Scatter(y=best_so_far, mode='lines',
                                      name='Лучший R2 до сих пор',
                                      line=dict(color='red', width=2)))
        fig_hist.update_layout(title="История оптимизации Optuna",
                               xaxis_title="Итерация", yaxis_title="R2",
                               template="plotly_white")
        st.plotly_chart(fig_hist, width='stretch')

        # Важность гиперпараметров
        try:
            importances = optuna.importance.get_param_importances(study)
            imp_df = pd.DataFrame(list(importances.items()),
                                  columns=['Параметр', 'Важность'])
            fig_imp = px.bar(imp_df.sort_values('Важность'),
                             x='Важность', y='Параметр', orientation='h',
                             title='Важность гиперпараметров (Optuna FAnova)',
                             template='plotly_white',
                             color='Важность',
                             color_continuous_scale='oranges')
            st.plotly_chart(fig_imp, width='stretch')
        except Exception:
            pass

# ── TAB 3: Early Stopping ────────────────────────────────────────────────────
with tab3:
    st.subheader("Early Stopping — предотвращение переобучения")
    st.markdown("""
    **Early Stopping** останавливает обучение, когда метрика на валидации
    перестаёт улучшаться. Это предотвращает переобучение и ускоряет обучение.
    """)

    es_rounds = st.slider("Patience (rounds без улучшения):", 5, 50, 20)

    if st.button("Обучить LightGBM с Early Stopping", type="primary", key="es"):
        log("Пользователь: нажата кнопка Early Stopping", patience=es_rounds)
        Xtr, Xval, ytr, yval = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42)

        with st.spinner("Обучение с Early Stopping (max 500 деревьев)..."):
            model_es = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, random_state=42,
                verbose=-1)
            model_es.fit(
                Xtr, ytr,
                eval_set=[(Xval, yval)],
                callbacks=[lgb.early_stopping(stopping_rounds=es_rounds,
                                               verbose=False),
                           lgb.log_evaluation(period=-1)]
            )

        best_iter = model_es.best_iteration_
        y_pred_es = model_es.predict(X_test)
        r2_es     = r2_score(y_test, y_pred_es)
        log("Early Stopping завершён", best_iter=best_iter,
            saved=500-best_iter, R2=round(r2_es, 4))

        c1, c2, c3 = st.columns(3)
        c1.metric("Итераций до остановки", best_iter)
        c2.metric("Итераций сохранено",    f"{500 - best_iter} ({(500-best_iter)/500*100:.0f}%)")
        c3.metric("R2 на тесте",            f"{r2_es:.4f}")

        # Кривая обучения
        results = model_es.evals_result_
        if 'valid_0' in results:
            val_losses = results['valid_0'].get(
                'l2', results['valid_0'].get('rmse', []))
            if val_losses:
                fig_es = go.Figure()
                fig_es.add_trace(go.Scatter(y=val_losses, mode='lines',
                                            name='Val L2 loss',
                                            line=dict(color='steelblue')))
                fig_es.add_vline(x=best_iter, line_dash='dash',
                                 line_color='red',
                                 annotation_text=f"Early Stop (iter={best_iter})")
                fig_es.update_layout(
                    title="Кривая обучения LightGBM с Early Stopping",
                    xaxis_title="Итерация", yaxis_title="Val Loss (L2)",
                    template="plotly_white")
                st.plotly_chart(fig_es, width='stretch')

        st.success(f"Early Stopping сэкономил {500-best_iter} лишних итераций!")
        st.info("Без Early Stopping модель продолжила бы переобучаться "
                "на обучающей выборке, теряя качество на тестовой.")
