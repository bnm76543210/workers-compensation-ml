"""
Задача 1: Сегментация данных — K-Means, DBSCAN, доменные правила
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess, feature_engineering
from src.logger import log
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Сегментация", layout="wide")
st.title("Сегментация данных и анализ по группам")
log("=== Страница 1: Сегментация загружена ===")

with st.spinner("Загрузка данных..."):
    df   = load_data()
    df_p = feature_engineering(preprocess(df))

target = 'UltimateIncurredClaimCost'
X = df_p.drop(columns=[target])
y = df_p[target]

tab1, tab2, tab3 = st.tabs(["K-Means", "DBSCAN", "Доменная сегментация"])

# ── TAB 1: K-Means ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Автоматическая сегментация через K-Means")

    n_clusters = st.slider("Количество сегментов:", 2, 6, 3)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X_scaled)
    df_p['Сегмент_KMeans'] = labels_km

    # Silhouette score
    sil = silhouette_score(X_scaled[:5000], labels_km[:5000])
    log("K-Means обучен", silhouette=round(sil, 4), n_clusters=st.session_state.get('n_clusters', '?'))
    st.metric("Silhouette Score (K-Means)", f"{sil:.3f}",
              help="От -1 до 1, чем выше — тем лучше разделены кластеры")

    # Статистика
    st.subheader("Статистика по K-Means сегментам")
    seg_stats = df_p.groupby('Сегмент_KMeans')[target].agg(
        ['count', 'mean', 'median', 'std']).round(2)
    seg_stats.columns = ['Записей', 'Среднее ($)', 'Медиана ($)', 'Std ($)']
    st.dataframe(seg_stats)

    # Boxplot
    fig_bx = px.box(df_p, x='Сегмент_KMeans', y=target,
                    title='Распределение выплат по K-Means сегментам',
                    labels={'Сегмент_KMeans': 'Сегмент',
                            target: 'Стоимость ($)'},
                    log_y=True, template='plotly_white',
                    color='Сегмент_KMeans',
                    color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bx, width='stretch')

    # Выбор числа кластеров (Elbow)
    st.subheader("Метод локтя — выбор оптимального K")
    if st.button("Построить Elbow-график", key="elbow"):
        with st.spinner("Вычисление WCSS..."):
            wcss = []
            ks = range(2, 9)
            for k in ks:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled[:10000])
                wcss.append(km.inertia_)
        fig_elbow = go.Figure(go.Scatter(x=list(ks), y=wcss, mode='lines+markers',
                                          line=dict(color='steelblue', width=2)))
        fig_elbow.update_layout(title='Метод локтя (WCSS by K)',
                                xaxis_title='K', yaxis_title='WCSS',
                                template='plotly_white')
        st.plotly_chart(fig_elbow, width='stretch')

    # Сравнение: общая vs сегментированные модели
    st.subheader("Общая модель vs Специализированные модели по сегментам")
    if st.button("Сравнить модели (K-Means)", type="primary", key="cmp_km"):
        results = {}
        X_all = X.copy()
        y_log = np.log1p(y)

        # Общая модель
        Xtr, Xte, ytr, yte = train_test_split(X_all, y_log,
                                                test_size=0.2, random_state=42)
        gen = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        gen.fit(Xtr, ytr)
        results['Общая модель'] = round(r2_score(yte, gen.predict(Xte)), 4)

        # Сегментированные
        r2_list = []
        for seg in range(n_clusters):
            mask = df_p['Сегмент_KMeans'] == seg
            Xs, ys = X_all[mask], y_log[mask]
            if len(Xs) < 100:
                continue
            Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
                Xs, ys, test_size=0.2, random_state=42)
            m = RandomForestRegressor(n_estimators=50, random_state=42,
                                      n_jobs=-1)
            m.fit(Xtr_s, ytr_s)
            r2_list.append(r2_score(yte_s, m.predict(Xte_s)))
        results['Сегментированные (среднее)'] = round(np.mean(r2_list), 4)

        comp_df = pd.DataFrame.from_dict(results, orient='index', columns=['R2'])
        st.dataframe(comp_df)

        fig_cmp = px.bar(comp_df.reset_index(), x='index', y='R2',
                         title='Сравнение: Общая vs Сегментированные модели',
                         labels={'index': 'Подход', 'R2': 'R2'},
                         color='R2', color_continuous_scale='greens',
                         template='plotly_white')
        fig_cmp.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_cmp, width='stretch')

        # Scatter предсказаний
        y_pred_gen = np.expm1(gen.predict(Xte))
        y_real_gen = np.expm1(yte)
        sample_i   = np.random.RandomState(42).choice(
            len(y_pred_gen), min(2000, len(y_pred_gen)), replace=False)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_real_gen.values[sample_i],
            y=y_pred_gen[sample_i],
            mode='markers', opacity=0.3,
            marker=dict(size=4, color='steelblue'),
            name='Общая модель'))
        max_v = float(max(y_real_gen.max(), y_pred_gen.max()))
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v], mode='lines',
            line=dict(color='red', dash='dash'), name='Идеал'))
        fig_scatter.update_layout(
            title='Факт vs Прогноз (общая модель, USD)',
            xaxis_title='Факт ($)', yaxis_title='Прогноз ($)',
            template='plotly_white', xaxis_type='log', yaxis_type='log')
        st.plotly_chart(fig_scatter, width='stretch')

# ── TAB 2: DBSCAN ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("DBSCAN — кластеризация на основе плотности")
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering) автоматически определяет
    количество кластеров и выявляет **выбросы** (шум).
    Не требует задавать число кластеров заранее.
    """)

    col1, col2 = st.columns(2)
    eps = col1.slider("eps (радиус окрестности):", 0.5, 5.0, 1.5, 0.1)
    min_samples = col2.slider("min_samples:", 5, 50, 10)

    if st.button("Запустить DBSCAN", type="primary", key="dbscan"):
        log("Пользователь: нажата кнопка DBSCAN")
        with st.spinner("DBSCAN на 5000 наблюдениях..."):
            # PCA до 10 компонент для скорости
            from sklearn.decomposition import PCA
            idx_db = np.random.RandomState(42).choice(
                len(X_scaled), min(5000, len(X_scaled)), replace=False)
            pca_db = PCA(n_components=10, random_state=42)
            X_pca_db = pca_db.fit_transform(X_scaled[idx_db])
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels_db = db.fit_predict(X_pca_db)

        n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_noise = (labels_db == -1).sum()
        c1, c2, c3 = st.columns(3)
        log("DBSCAN завершён", clusters=n_clusters_db, noise=n_noise,
            noise_pct=round(n_noise/len(labels_db)*100, 1))
        c1.metric("Найдено кластеров", n_clusters_db)
        c2.metric("Выбросов (шум)", f"{n_noise} ({n_noise/len(labels_db)*100:.1f}%)")
        c3.metric("Наблюдений", len(labels_db))

        if n_clusters_db > 0:
            # PCA 2D для визуализации
            pca2d = PCA(n_components=2, random_state=42)
            X2d = pca2d.fit_transform(X_pca_db)
            plot_df = pd.DataFrame({
                'PC1': X2d[:, 0], 'PC2': X2d[:, 1],
                'Кластер': labels_db.astype(str),
                'Стоимость': np.log1p(y.values[idx_db])
            })
            fig_db = px.scatter(plot_df, x='PC1', y='PC2',
                                color='Кластер', opacity=0.5,
                                title=f'DBSCAN: {n_clusters_db} кластеров '
                                      f'+ {n_noise} выбросов (eps={eps})',
                                template='plotly_white',
                                color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig_db, width='stretch')

            # Статистика по кластерам
            noise_label = '-1'
            valid_df = plot_df[plot_df['Кластер'] != noise_label].copy()
            if not valid_df.empty:
                seg_stat_db = (valid_df.groupby('Кластер')['Стоимость']
                               .agg(['count', 'mean']).round(3))
                seg_stat_db.columns = ['Записей', 'Среднее log(стоимость)']
                st.dataframe(seg_stat_db)
        else:
            st.warning("DBSCAN не нашёл кластеров. Попробуйте увеличить eps.")

# ── TAB 3: Доменная сегментация ───────────────────────────────────────────────
with tab3:
    st.subheader("Сегментация по доменным знаниям")
    st.markdown("""
    Разбиение по **бизнес-правилам** страховой сферы — более интерпретируемо,
    чем автоматические алгоритмы.
    """)

    # Правила сегментации
    df_domain = df_p.copy()
    cost = df_domain[target]
    age  = df_domain['Age'] if 'Age' in df_domain.columns else pd.Series(0, index=df_domain.index)
    pay  = df_domain['WeeklyPay'] if 'WeeklyPay' in df_domain.columns else pd.Series(0, index=df_domain.index)
    est  = df_domain['InitialCaseEstimate'] if 'InitialCaseEstimate' in df_domain.columns else pd.Series(0, index=df_domain.index)

    def assign_segment(row):
        init = row.get('InitialCaseEstimate', 0)
        if init < 1000:
            return 'Лёгкий случай'
        elif init < 10000:
            return 'Средний случай'
        else:
            return 'Тяжёлый случай'

    df_domain['Сегмент_Domain'] = df_domain.apply(assign_segment, axis=1)

    seg_order = ['Лёгкий случай', 'Средний случай', 'Тяжёлый случай']
    seg_colors = {'Лёгкий случай': '#4CAF50',
                  'Средний случай': '#FF9800',
                  'Тяжёлый случай': '#F44336'}

    st.markdown("**Правила сегментации:**")
    st.markdown("- **Лёгкий случай**: InitialCaseEstimate < $1 000")
    st.markdown("- **Средний случай**: $1 000 ≤ InitialCaseEstimate < $10 000")
    st.markdown("- **Тяжёлый случай**: InitialCaseEstimate ≥ $10 000")

    domain_stats = df_domain.groupby('Сегмент_Domain')[target].agg(
        ['count', 'mean', 'median']).round(2)
    domain_stats.columns = ['Записей', 'Среднее ($)', 'Медиана ($)']
    st.dataframe(domain_stats)

    fig_dom = px.box(df_domain, x='Сегмент_Domain', y=target,
                     title='Распределение выплат по доменным сегментам',
                     labels={'Сегмент_Domain': 'Сегмент',
                             target: 'Стоимость ($)'},
                     log_y=True, template='plotly_white',
                     category_orders={'Сегмент_Domain': seg_order},
                     color='Сегмент_Domain', color_discrete_map=seg_colors)
    st.plotly_chart(fig_dom, width='stretch')

    # Сравнение моделей по доменным сегментам
    if st.button("Сравнить модели (доменная сегментация)",
                 type="primary", key="cmp_domain"):
        y_log = np.log1p(y)
        results_d = {}

        Xtr, Xte, ytr, yte = train_test_split(X, y_log,
                                                test_size=0.2, random_state=42)
        gen = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        gen.fit(Xtr, ytr)
        results_d['Общая модель'] = round(r2_score(yte, gen.predict(Xte)), 4)

        r2_dom = []
        for seg in seg_order:
            mask = df_domain['Сегмент_Domain'] == seg
            Xs, ys = X[mask], y_log[mask]
            if len(Xs) < 100:
                continue
            Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
                Xs, ys, test_size=0.2, random_state=42)
            m = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            m.fit(Xtr_s, ytr_s)
            r2_dom.append(r2_score(yte_s, m.predict(Xte_s)))
        results_d['Доменные модели (среднее)'] = round(np.mean(r2_dom), 4)

        comp_d = pd.DataFrame.from_dict(results_d, orient='index', columns=['R2'])
        st.dataframe(comp_d)

        fig_cmp_d = px.bar(comp_d.reset_index(), x='index', y='R2',
                            title='Общая vs Доменная сегментация',
                            labels={'index': 'Подход'},
                            color='R2', color_continuous_scale='blues',
                            template='plotly_white')
        fig_cmp_d.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_cmp_d, width='stretch')
