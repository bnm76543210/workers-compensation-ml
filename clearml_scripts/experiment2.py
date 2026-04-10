"""
ClearML: Эксперимент 2 — XGBoost с Optuna
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from clearml import Task, Dataset
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

task   = Task.init(project_name="Workers Compensation",
                   task_name="Experiment 2 — XGBoost + Optuna",
                   task_type=Task.TaskTypes.training)
logger = task.get_logger()

print("Загрузка датасета из ClearML...")
dataset    = Dataset.get(dataset_name="Workers Compensation Dataset",
                         dataset_project="Workers Compensation")
local_path = dataset.get_local_copy()
df = pd.read_csv(os.path.join(local_path, "workers_comp_raw.csv"))

target = 'UltimateIncurredClaimCost'
drop_cols = [target] + (['ClaimDescription']
             if 'ClaimDescription' in df.columns else [])
X = df.drop(columns=drop_cols)
y = pd.to_numeric(df[target], errors='coerce')

for col in X.select_dtypes(include=['object','category']).columns:
    X[col] = LabelEncoder().fit_transform(
        X[col].fillna('Unknown').astype(str))
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())

mask = y.notnull()
X, y = X[mask], np.log1p(y[mask])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optuna
def objective(trial):
    p = {
        'n_estimators':     trial.suggest_int('n_estimators', 50, 300),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'learning_rate':    trial.suggest_float('learning_rate',
                                                 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                 0.6, 1.0),
        'random_state': 42, 'verbosity': 0
    }
    return cross_val_score(xgb.XGBRegressor(**p),
                           X_train, y_train, cv=3,
                           scoring='r2').mean()

print("Запуск Optuna (30 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
print(f"Лучший R² (CV): {study.best_value:.4f}")
print(f"Лучшие параметры: {best_params}")

task.connect(best_params)

model  = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

logger.report_scalar("Metrics", "R2",   iteration=0, value=r2)
logger.report_scalar("Metrics", "RMSE", iteration=0, value=rmse)
logger.report_scalar("Metrics", "MAE",  iteration=0, value=mae)

# Сравнение с baseline
logger.report_table(
    title="Сравнение моделей", series="Comparison", iteration=0,
    table_plot=pd.DataFrame({
        "Модель":           ["Linear Regression", "XGBoost + Optuna"],
        "R²":               [None, round(r2, 4)],
        "RMSE":             [None, round(rmse, 4)],
        "MAE":              [None, round(mae, 4)],
    })
)

# Feature Importance
feat_imp = pd.Series(model.feature_importances_,
                     index=X.columns).sort_values(ascending=False).head(10)
fig, ax  = plt.subplots(figsize=(10, 6))
ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color='steelblue')
ax.set_xlabel("Важность")
ax.set_title("Топ-10 важных признаков (XGBoost)")
plt.tight_layout()
logger.report_matplotlib_figure("Feature Importance", "importance", 0, fig)
plt.close()

# Optuna история
fig, ax = plt.subplots(figsize=(10, 4))
vals = [t.value for t in study.trials]
ax.plot(vals, color='steelblue', linewidth=1.5)
ax.axhline(study.best_value, color='red', linestyle='--',
           label=f'Лучший R²={study.best_value:.4f}')
ax.set_xlabel("Итерация"); ax.set_ylabel("R²")
ax.set_title("История оптимизации Optuna"); ax.legend()
plt.tight_layout()
logger.report_matplotlib_figure("Optuna History", "optuna", 0, fig)
plt.close()

task.close()
print("✅ Эксперимент 2 завершён!")