"""
ClearML Serving preprocessing script.
Transforms raw input JSON into model-ready feature vector.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_ORDER = [
    'Age', 'Gender', 'MaritalStatus', 'DependentChildren', 'DependentsOther',
    'WeeklyPay', 'PartTimeFullTime', 'HoursWorkedPerWeek', 'DaysWorkedPerWeek',
    'InitialCaseEstimate', 'Accident_Year', 'Accident_Month',
    'Accident_DayOfWeek', 'Reported_Year', 'Reported_Month',
    'Reported_DayOfWeek', 'ReportDelay_Days', 'Age_x_WeeklyPay',
    'Estimate_per_Pay', 'HasDependents', 'IsFullTime', 'Log_InitialEstimate',
]


def preprocess(data: dict) -> dict:
    """
    Input: raw dict from HTTP request
    Output: dict with 'input' key containing feature array
    """
    df = pd.DataFrame([data])

    # Encode categorical
    for col in ['Gender', 'MaritalStatus', 'PartTimeFullTime']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(
                df[col].fillna('Unknown').astype(str))

    # Feature engineering
    age  = float(df.get('Age', [35])[0])
    pay  = float(df.get('WeeklyPay', [500])[0])
    est  = float(df.get('InitialCaseEstimate', [5000])[0])
    hrs  = float(df.get('HoursWorkedPerWeek', [40])[0])
    dep  = float(df.get('DependentsOther', [0])[0])

    df['Age_x_WeeklyPay']    = age * pay
    df['Estimate_per_Pay']   = est / (pay + 1)
    df['HasDependents']      = int(dep > 0)
    df['IsFullTime']         = int(hrs >= 35)
    df['Log_InitialEstimate'] = np.log1p(est)

    # Fill missing cols
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    return {"input": df[FEATURE_ORDER].values.tolist()}


def postprocess(data: dict) -> dict:
    """Convert log-scale prediction back to USD."""
    pred_log = data.get("prediction", [0])
    if isinstance(pred_log, list):
        pred_usd = float(np.expm1(pred_log[0]))
    else:
        pred_usd = float(np.expm1(pred_log))
    return {"predicted_cost_usd": round(pred_usd, 2)}
