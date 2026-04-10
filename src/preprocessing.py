"""
Предобработка и Feature Engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Числовые признаки ───────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = 'UltimateIncurredClaimCost'
    if target in numeric_cols:
        numeric_cols.remove(target)

    # Заполняем медианой
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # ── Дата → числовые признаки ────────────────────────────────────────────
    date_cols = ['DateTimeOfAccident', 'DateReported']
    for dcol in date_cols:
        if dcol in df.columns:
            parsed = pd.to_datetime(df[dcol], errors='coerce')
            prefix = 'Accident' if 'Accident' in dcol else 'Reported'
            df[f'{prefix}_Year']    = parsed.dt.year.fillna(0).astype(int)
            df[f'{prefix}_Month']   = parsed.dt.month.fillna(0).astype(int)
            df[f'{prefix}_DayOfWeek'] = parsed.dt.dayofweek.fillna(0).astype(int)
            df = df.drop(columns=[dcol])

    # Признак задержки сообщения (в днях)
    if 'Accident_Year' in df.columns and 'Reported_Year' in df.columns:
        df['ReportDelay_Days'] = (
            df['Reported_Year'] * 365 + df['Reported_Month'] * 30
            - df['Accident_Year'] * 365 - df['Accident_Month'] * 30
        ).clip(lower=0)

    # ── Категориальные признаки ─────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    skip = ['ClaimDescription']
    for col in cat_cols:
        if col in skip:
            continue
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Удаляем текстовый столбец
    if 'ClaimDescription' in df.columns:
        df = df.drop(columns=['ClaimDescription'])

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Признаки взаимодействия ─────────────────────────────────────────────
    if 'Age' in df.columns and 'WeeklyPay' in df.columns:
        df['Age_x_WeeklyPay'] = df['Age'] * df['WeeklyPay']

    if 'InitialCaseEstimate' in df.columns and 'WeeklyPay' in df.columns:
        df['Estimate_per_Pay'] = df['InitialCaseEstimate'] / \
                                 (df['WeeklyPay'] + 1)

    # ── Бинарные признаки ───────────────────────────────────────────────────
    dep_col = next(
        (c for c in ['DependentsOther', 'DependentsOtherThanSpouse',
                     'DependentChildren']
         if c in df.columns), None
    )
    if dep_col:
        df['HasDependents'] = (pd.to_numeric(df[dep_col],
                                              errors='coerce').fillna(0) > 0
                               ).astype(int)

    if 'HoursWorkedPerWeek' in df.columns:
        df['IsFullTime'] = (df['HoursWorkedPerWeek'] >= 35).astype(int)

    # ── Лог-преобразование целевой переменной (если есть) ───────────────────
    if 'InitialCaseEstimate' in df.columns:
        df['Log_InitialEstimate'] = np.log1p(df['InitialCaseEstimate'])

    return df