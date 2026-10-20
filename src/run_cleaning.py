"""Run data cleaning end-to-end and save processed data + report artefacts."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clean import NUM_COLS, clean_test, clean_train, missing_report, outlier_report


def _read_raw(path: str) -> pd.DataFrame:
    """Read a raw CSV with minimal guardrails for malformed data files."""
    df = pd.read_csv(path)
    return df.copy()


def main() -> None:
    # ---- paths ----------------------------------------------------------
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW = os.path.join(ROOT, 'data', 'raw')
    PROC = os.path.join(ROOT, 'data', 'processed')
    REP = os.path.join(ROOT, 'report')
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(REP, exist_ok=True)

    # ---- load raw for diagnostics --------------------------------------
    train_raw = _read_raw(os.path.join(RAW, 'train_u6lujuX_CVtuZ9i.csv'))
    test_raw = _read_raw(os.path.join(RAW, 'test_Y3wMUE5_7gLdaTN.csv'))

    # ---- report tables --------------------------------------------------
    print("\n=== TRAIN — missing values ===")
    train_missing = missing_report(train_raw)
    print(train_missing)
    train_missing.to_csv(os.path.join(REP, 'table_missing_train.csv'))

    print("\n=== TEST — missing values ===")
    test_missing = missing_report(test_raw)
    print(test_missing)
    test_missing.to_csv(os.path.join(REP, 'table_missing_test.csv'))

    print("\n=== TRAIN — IQR outliers ===")
    train_outliers = outlier_report(train_raw)
    print(train_outliers.round(2))
    train_outliers.to_csv(os.path.join(REP, 'table_outliers_train.csv'), index=False)

    # ---- clean ----------------------------------------------------------
    train, cap = clean_train(os.path.join(RAW, 'train_u6lujuX_CVtuZ9i.csv'))
    test = clean_test(os.path.join(RAW, 'test_Y3wMUE5_7gLdaTN.csv'), cap)
    if not np.isfinite(cap):
        cap = float(train['LoanAmount'].dropna().quantile(0.99)) if train['LoanAmount'].notna().any() else 128.0
    print(f"\nLoanAmount 99th-percentile cap (train): {cap:.1f}")

    # ---- save processed -------------------------------------------------
    train.to_csv(os.path.join(PROC, 'train_clean.csv'), index=False)
    test.to_csv(os.path.join(PROC, 'test_clean.csv'), index=False)
    print(f"\nSaved: train_clean {train.shape}, test_clean {test.shape}")

    # ---- figures --------------------------------------------------------
    sns.set_style('whitegrid')

    # 1. Missing-value heatmap (train)
    plt.figure(figsize=(10, 5))
    sns.heatmap(train_raw.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values — Training Set')
    plt.tight_layout()
    plt.savefig(os.path.join(REP, 'fig_missing_heatmap.png'), dpi=150)
    plt.close()

    # 2. Outlier boxplots (before)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), NUM_COLS):
        values = train_raw[col].dropna()
        if values.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            sns.boxplot(x=values, ax=ax)
        ax.set_title(f'{col} (before)')
    plt.tight_layout()
    plt.savefig(os.path.join(REP, 'fig_outlier_boxplots.png'), dpi=150)
    plt.close()

    # 3. Distributions before/after log1p
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [('ApplicantIncome', 'ApplicantIncome_log'),
             ('CoapplicantIncome', 'CoapplicantIncome_log'),
             ('LoanAmount', 'LoanAmount_log')]
    for ax, (raw, log) in zip(axes, pairs):
        values = train[log].dropna()
        if values.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            sns.histplot(values, kde=True, ax=ax, color='steelblue')
        ax.set_title(f'{raw} (log1p)')
    plt.tight_layout()
    plt.savefig(os.path.join(REP, 'fig_log_transformed.png'), dpi=150)
    plt.close()

    # 4. Credit history vs outcome — strongest predictor insight
    if 'Credit_History' in train_raw.columns and 'Loan_Status' in train_raw.columns:
        ct = pd.crosstab(train_raw['Credit_History'], train_raw['Loan_Status'], normalize='index')
        ct.plot(kind='bar', figsize=(6, 4), color=['#d62728', '#2ca02c'])
        plt.title('Loan Approval Rate by Credit History')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(REP, 'fig_credit_history.png'), dpi=150)
        plt.close()

    # 5. Correlation heatmap
    num = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History']
    num = [c for c in num if c in train_raw.columns]
    if num:
        corr_data = train_raw[num].apply(pd.to_numeric, errors='coerce')
        corr_data = corr_data.loc[:, corr_data.notna().any()]  # drop all-NaN columns
        if corr_data.shape[1] > 1:
            corr = corr_data.corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title('Correlation Matrix (raw numerics)')
            plt.tight_layout()
            plt.savefig(os.path.join(REP, 'fig_correlation.png'), dpi=150)
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No usable numeric data for correlation', ha='center', va='center')
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(os.path.join(REP, 'fig_correlation.png'), dpi=150)
            plt.close(fig)

    print("\nDone. Check data/processed/ and report/ for outputs.")


if __name__ == '__main__':
    main()