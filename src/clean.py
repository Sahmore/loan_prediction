"""Data cleaning for the loan prediction project.

All functions are pure: take a DataFrame, return a cleaned DataFrame.
Fitted statistics (e.g. the LoanAmount 99th-percentile cap) are learned
from the training set only and passed explicitly to the test set to
avoid data leakage.

Design principle: this module does NOT impute. Missing values are
preserved so that the sklearn pipeline in model_creation.py can fit
imputers on training data only.
"""
import numpy as np
import pandas as pd

NUM_COLS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']


# ---------------------------------------------------------------- diagnostics
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Table of missing values per column (for the report)."""
    m = df.isnull().sum()
    pct = (m / len(df) * 100).round(2)
    return (pd.DataFrame({'Missing': m, 'Percent': pct})
            .query('Missing > 0')
            .sort_values('Missing', ascending=False))


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """IQR outlier counts per numeric column (for the report)."""
    rows = []
    for c in NUM_COLS:
        if c not in df.columns:
            continue
        numeric = pd.to_numeric(df[c], errors='coerce')
        if numeric.dropna().empty:
            continue
        q1, q3 = numeric.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((numeric < lo) | (numeric > hi)).sum())
        rows.append({'Column': c, 'Q1': q1, 'Q3': q3,
                     'Lower': lo, 'Upper': hi,
                     'Outliers': n, 'Percent': round(n / len(df) * 100, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cleaning
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise Dependents ('3+' -> 3) and force numeric columns.

    Missing values are preserved as NaN — imputation happens inside the
    sklearn pipeline so that train statistics never leak into test.
    """
    df = df.copy()
    if 'Dependents' in df.columns:
        df['Dependents'] = (df['Dependents']
                            .astype(str).str.replace('3+', '3', regex=False)
                            .replace('nan', np.nan).astype(float))
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')  # NaN preserved
    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform right-skewed income/loan features.

    np.log1p(NaN) = NaN, so missing values stay missing for the pipeline.
    """
    df = df.copy()
    for c, log_name in [('ApplicantIncome', 'ApplicantIncome_log'),
                        ('CoapplicantIncome', 'CoapplicantIncome_log'),
                        ('LoanAmount', 'LoanAmount_log')]:
        if c in df.columns:
            df[log_name] = np.log1p(df[c])   # NaN preserved
    return df


def cap_loan_amount(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Winsorize LoanAmount at a cap learned from the training set.

    The cap must be a finite number learned from train — passing anything
    else raises, rather than silently falling back to a per-file statistic
    (which would leak test information into the transform).
    """
    if not np.isfinite(cap):
        raise ValueError("loan_amount_cap must be a finite number learned from train")
    df = df.copy()
    df['LoanAmount_capped'] = df['LoanAmount'].clip(upper=cap)
    return df


def clean_train(path: str):
    """Load and clean the training set.

    Returns
    -------
    (cleaned_df, loan_amount_cap)
        cleaned_df      : training DataFrame with engineered columns and NaN preserved
        loan_amount_cap : 99th percentile of LoanAmount on train (used for test)
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=['Loan_Status'])      # target must exist
    df = fix_dtypes(df)

    # Compute the cap on the raw (pre-log) LoanAmount BEFORE adding log features
    # so the cap is in the same units as the values it will clip.
    numeric = df['LoanAmount'].dropna()
    if numeric.empty:
        raise ValueError("LoanAmount column is entirely empty in training data")
    cap = float(numeric.quantile(0.99))

    df = add_log_features(df)
    df = cap_loan_amount(df, cap)

    return df, cap


def clean_test(path: str, loan_amount_cap: float) -> pd.DataFrame:
    """Load and clean the test set using the train-learned cap."""
    df = pd.read_csv(path)
    df = fix_dtypes(df)
    df = add_log_features(df)
    df = cap_loan_amount(df, loan_amount_cap)
    return df