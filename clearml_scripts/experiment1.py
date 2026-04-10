"""
ClearML: Эксперимент 1 — Linear Regression (baseline)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from clearml import Task, Dataset
import warnings
warnings.filterwarnings('ignore')

task   = Task.init(project_name="Workers Compensation",
                   task_name="Experiment 1 — Linear Regression Baseline",
                   task_type=Task.TaskTypes.training)
logger = task.get_logger()

print("Загрузка датасета из ClearML...")
dataset    = Dataset.get(dataset_name="Workers Compensation Dataset",
                         dataset_project="Workers Compensation")
local_path = dataset.get_local_copy()
df = pd.read_csv(os.path.join(local_path, "workers_comp_raw.csv"))
print(f"Загружено: {len(df)} строк")

# Предобработка
target = 'UltimateIncurredClaimCost'
drop_cols = [target]
if 'ClaimDescription' in df.columns:
    drop_cols.append('ClaimDescription')
X = df.drop(columns=drop_cols)
y = pd.to_numeric(df[target], errors='coerce')

for col in X.select_dtypes(include=['object','category']).columns:
    X[col] = X[col].fillna('Unknown')
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())

mask = y.notnull()
X, y = X[mask], np.log1p(y[mask])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

params = {"model": "LinearRegression", "scaling": "StandardScaler",
          "test_size": 0.2, "target_transform": "log1p"}
task.connect(params)

model  = LinearRegression()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

logger.report_scalar("Metrics", "R2",   iteration=0, value=r2)
logger.report_scalar("Metrics", "RMSE", iteration=0, value=rmse)
logger.report_scalar("Metrics", "MAE",  iteration=0, value=mae)

# Scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.4, s=15, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Идеальный прогноз')
ax.set_xlabel("Факт (log)"); ax.set_ylabel("Прогноз (log)")
ax.set_title("Linear Regression: Факт vs Прогноз")
ax.legend(); plt.tight_layout()
logger.report_matplotlib_figure("Факт vs Прогноз", "scatter", 0, fig)
plt.close()

# Residuals
residuals = y_test.values - y_pred
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_pred, residuals, alpha=0.4, s=15, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel("Прогноз"); ax.set_ylabel("Остатки")
ax.set_title("Residuals Plot"); plt.tight_layout()
logger.report_matplotlib_figure("Residuals", "residuals", 0, fig)
plt.close()

task.close()
print("Experiment 1 done!")