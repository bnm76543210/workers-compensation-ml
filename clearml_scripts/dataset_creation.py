"""
ClearML: Создание датасета Workers Compensation
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from clearml import Dataset
import os
import warnings
warnings.filterwarnings('ignore')

print("Загрузка Workers Compensation Dataset...")
data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
df   = data.frame.copy()

print(f"Записей: {len(df)}, Признаков: {df.shape[1]}")
print(f"\nПервые 5 строк:\n{df.head()}")

os.makedirs("data", exist_ok=True)
csv_path = "data/workers_comp_raw.csv"
df.to_csv(csv_path, index=False)
print(f"Сохранено: {csv_path}")

dataset = Dataset.create(
    dataset_name="Workers Compensation Dataset",
    dataset_project="Workers Compensation",
    dataset_tags=["insurance", "regression", "workers-comp"],
    description=(
        "Workers Compensation Dataset (OpenML ID: 42876). "
        "Задача: регрессия — прогнозирование итоговой стоимости "
        "страхового возмещения (UltimateIncurredClaimCost)."
    )
)

dataset.add_files(path=csv_path)

y = pd.to_numeric(df['UltimateIncurredClaimCost'], errors='coerce')
dataset.get_logger().report_text(
    f"=== Workers Compensation Dataset ===\n"
    f"Записей:             {len(df)}\n"
    f"Признаков:           {df.shape[1]}\n"
    f"Пропусков:           {df.isnull().sum().sum()}\n\n"
    f"=== Целевая переменная: UltimateIncurredClaimCost ===\n"
    f"Среднее:   ${y.mean():,.2f}\n"
    f"Медиана:   ${y.median():,.2f}\n"
    f"Мин:       ${y.min():,.2f}\n"
    f"Макс:      ${y.max():,.2f}\n"
    f"Std:       ${y.std():,.2f}"
)

num_cols = df.select_dtypes(include=[np.number]).columns[:6].tolist()
stats = df[num_cols].describe().round(2)
dataset.get_logger().report_table(
    title="Описательная статистика",
    series="Workers Comp Stats",
    table_plot=stats
)

dataset.finalize(auto_upload=True)
print(f"\nДатасет создан! ID: {dataset.id}")