"""
Вспомогательные функции для интеграции с ClearML.
Используется clearml_scripts/ и страницей pages/5_ClearML.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clearml import Task, Dataset
import os


# ── Инициализация задачи ─────────────────────────────────────────────────────
def init_task(project_name: str,
              task_name: str,
              task_type=None) -> Task:
    """
    Инициализирует ClearML Task.
    Возвращает объект task для дальнейшего логирования.
    """
    if task_type is None:
        task_type = Task.TaskTypes.training
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type
    )
    return task


# ── Логирование метрик ───────────────────────────────────────────────────────
def log_metrics(task: Task, metrics: dict, iteration: int = 0):
    """
    Логирует словарь метрик в ClearML.
    Пример: log_metrics(task, {"R2": 0.85, "RMSE": 0.12})
    """
    logger = task.get_logger()
    for metric_name, value in metrics.items():
        logger.report_scalar(
            title="Metrics",
            series=metric_name,
            value=value,
            iteration=iteration
        )


# ── Логирование гиперпараметров ──────────────────────────────────────────────
def log_params(task: Task, params: dict):
    """Логирует гиперпараметры модели в ClearML."""
    task.connect(params)


# ── Логирование графика ──────────────────────────────────────────────────────
def log_figure(task: Task, fig: plt.Figure,
               title: str, series: str, iteration: int = 0):
    """Логирует matplotlib-фигуру в ClearML Plots."""
    logger = task.get_logger()
    logger.report_matplotlib_figure(
        title=title,
        series=series,
        figure=fig,
        iteration=iteration
    )
    plt.close(fig)


# ── Логирование таблицы ──────────────────────────────────────────────────────
def log_table(task: Task, df: pd.DataFrame,
              title: str, series: str, iteration: int = 0):
    """Логирует pandas DataFrame как таблицу в ClearML."""
    logger = task.get_logger()
    logger.report_table(
        title=title,
        series=series,
        table_plot=df,
        iteration=iteration
    )


# ── Стандартные графики для регрессии ───────────────────────────────────────
def log_regression_plots(task: Task,
                          y_test, y_pred,
                          model_name: str = "Model"):
    """
    Логирует стандартный набор графиков для задачи регрессии:
    - Scatter: факт vs прогноз
    - Residuals plot
    - Гистограмма ошибок
    """
    logger = task.get_logger()

    # 1. Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.4, s=15, color='steelblue')
    lims = [min(float(np.min(y_test)), float(np.min(y_pred))),
            max(float(np.max(y_test)), float(np.max(y_pred)))]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Идеальный прогноз')
    ax.set_xlabel("Фактическое значение")
    ax.set_ylabel("Предсказанное значение")
    ax.set_title(f"{model_name}: Факт vs Прогноз")
    ax.legend()
    plt.tight_layout()
    logger.report_matplotlib_figure("Факт vs Прогноз", "scatter", 0, fig)
    plt.close()

    # 2. Residuals
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Предсказанное значение")
    ax.set_ylabel("Остатки")
    ax.set_title(f"{model_name}: График остатков")
    plt.tight_layout()
    logger.report_matplotlib_figure("Residuals", "residuals", 0, fig)
    plt.close()

    # 3. Гистограмма остатков
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color='steelblue', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Остатки")
    ax.set_ylabel("Количество")
    ax.set_title(f"{model_name}: Распределение остатков")
    plt.tight_layout()
    logger.report_matplotlib_figure("Распределение остатков",
                                     "hist", 0, fig)
    plt.close()


# ── Получение датасета из ClearML ────────────────────────────────────────────
def get_dataset(dataset_name: str,
                dataset_project: str,
                csv_filename: str) -> pd.DataFrame:
    """
    Загружает датасет из ClearML и возвращает DataFrame.
    Пример:
        df = get_dataset("Workers Compensation Dataset",
                         "Workers Compensation",
                         "workers_comp_raw.csv")
    """
    dataset    = Dataset.get(dataset_name=dataset_name,
                              dataset_project=dataset_project)
    local_path = dataset.get_local_copy()
    return pd.read_csv(os.path.join(local_path, csv_filename))