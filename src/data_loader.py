"""
Загрузка и кэширование датасета Workers Compensation
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import streamlit as st

@st.cache_data
def load_data():
    data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
    df = data.frame.copy()

    # Приводим числовые столбцы
    numeric_cols = ['Age', 'WeeklyPay', 'InitialCaseEstimate',
                    'UltimateIncurredClaimCost', 'HoursWorkedPerWeek',
                    'DaysWorkedPerWeek']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def get_feature_target(df):
    target = 'UltimateIncurredClaimCost'
    drop_cols = [target, 'ClaimDescription'] \
                if 'ClaimDescription' in df.columns else [target]
    X = df.drop(columns=drop_cols)
    y = df[target]
    return X, y