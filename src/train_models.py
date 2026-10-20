"""Train, tune, and evaluate loan approval classifiers.

Reads data/processed/train_clean.csv and produces:
  - models/final_model.pkl           (full pipeline: preprocessor + classifier)
  - report/model_comparison_untuned.csv
  - report/model_comparison_tuned.csv
  - report/final_metrics.txt
  - report/fig_confusion_matrix.png
  - report/fig_model_comparison.png
  - report/fig_roc_curve.png
  - data/processed/submission.csv   (predictions on the 367-row test set)
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, 'data', 'processed')
MOD  = os.path.join(ROOT, 'models')
REP  = os.path.join(ROOT, 'report')
os.makedirs(MOD, exist_ok=True)
os.makedirs(REP, exist_ok=True)

# ---------------------------------------------------------------- feature groups
NUM_COLS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'ApplicantIncome_log',
            'CoapplicantIncome_log', 'LoanAmount_log', 'LoanAmount_capped']
CAT_COLS = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area', 'Credit_History']


def load_train():
    df = pd.read_csv(os.path.join(PROC, 'train_clean.csv'))
    y  = df['Loan_Status'].map({'Y': 1, 'N': 0}).astype(int)
    # Keep only columns used by the model (drop Loan_ID and target)
    X  = df.drop(columns=['Loan_ID', 'Loan_Status'])
    return X, y


def build_preprocessor():
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, NUM_COLS),
        ('cat', cat_pipe, CAT_COLS),
    ])


# ---------------------------------------------------------------- main
def main():
    X, y = load_train()
    print(f"Loaded training data: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(X_train)} | Holdout: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    candidates = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree':       DecisionTreeClassifier(random_state=42),
        'RandomForest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'KNN':                KNeighborsClassifier(),
    }

    # ---------- Task 4: cross-validated training (untuned) ----------
    untuned_rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([('pre', build_preprocessor()), ('clf', clf)])
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        untuned_rows.append({
            'Model': name,
            **{m: round(res[f'test_{m}'].mean(), 4) for m in scoring},
            'acc_std': round(res['test_accuracy'].std(), 4),
        })
        print(f"[CV] {name:20s} acc={res['test_accuracy'].mean():.4f} "
              f"auc={res['test_roc_auc'].mean():.4f}")

    untuned_df = pd.DataFrame(untuned_rows).sort_values('roc_auc', ascending=False)
    untuned_df.to_csv(os.path.join(REP, 'model_comparison_untuned.csv'), index=False)
    print("\nUntuned comparison:")
    print(untuned_df.to_string(index=False))

    # ---------- Task 5: GridSearch top 2 by CV ROC-AUC ----------
    top2 = untuned_df.head(2)['Model'].tolist()
    print(f"\nTuning top 2: {top2}")

    grids = {
        'RandomForest': {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5, 10],
            'clf__class_weight': [None, 'balanced'],
        },
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__class_weight': [None, 'balanced'],
        },
        'DecisionTree': {
            'clf__max_depth': [3, 5, 10, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__class_weight': [None, 'balanced'],
        },
        'KNN': {
            'clf__n_neighbors': [3, 5, 7, 11, 15],
            'clf__weights': ['uniform', 'distance'],
        },
    }

    tuned_rows, best_models = [], {}
    for name in top2:
        pipe = Pipeline([('pre', build_preprocessor()), ('clf', candidates[name])])
        gs = GridSearchCV(pipe, grids[name], cv=cv, scoring='roc_auc',
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_
        tuned_rows.append({
            'Model': name,
            'cv_roc_auc': round(gs.best_score_, 4),
            'best_params': str(gs.best_params_),
        })
        print(f"[TUNED] {name:20s} auc={gs.best_score_:.4f} params={gs.best_params_}")

    tuned_df = pd.DataFrame(tuned_rows).sort_values('cv_roc_auc', ascending=False)
    tuned_df.to_csv(os.path.join(REP, 'model_comparison_tuned.csv'), index=False)

    # ---------- Final evaluation on holdout ----------
    final_name = tuned_df.iloc[0]['Model']
    final = best_models[final_name]
    y_pred  = final.predict(X_test)
    y_proba = final.predict_proba(X_test)[:, 1]

    metrics = {
        'model': final_name,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
        'f1':        round(f1_score(y_test, y_pred), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n=== FINAL: {final_name} (holdout) ===")
    print(metrics)
    print(classification_report(y_test, y_pred, target_names=['N (rejected)', 'Y (approved)']))

    with open(os.path.join(REP, 'final_metrics.txt'), 'w') as f:
        f.write(f"Final model: {final_name}\n")
        for k, v in metrics.items():
            if k != 'model':
                f.write(f"{k}: {v}\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(y_test, y_pred,
                                      target_names=['N (rejected)', 'Y (approved)']))

    # ---------- Save model ----------
    joblib.dump(final, os.path.join(MOD, 'final_model.pkl'))
    with open(os.path.join(MOD, 'final_model_name.txt'), 'w') as f:
        f.write(final_name)
    print(f"Saved model -> {os.path.join(MOD, 'final_model.pkl')}")

    # ---------- Figures ----------
    sns.set_style('whitegrid')

    # Fig: confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'Confusion Matrix — {final_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(REP, 'fig_confusion_matrix.png'), dpi=150)
    plt.close()

    # Fig: model comparison (untuned)
    plot_df = untuned_df.set_index('Model')[['accuracy', 'roc_auc', 'f1']]
    plot_df.plot(kind='bar', figsize=(9, 4), rot=0)
    plt.title('Model comparison (untuned, 5-fold CV)')
    plt.ylabel('Score'); plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(REP, 'fig_model_comparison.png'), dpi=150)
    plt.close()

    # Fig: ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'{final_name} (AUC={metrics["roc_auc"]:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (holdout)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(REP, 'fig_roc_curve.png'), dpi=150)
    plt.close()

    # ---------- Predict on 367-row test set ----------
    test = pd.read_csv(os.path.join(PROC, 'test_clean.csv'))
    test_X = test.drop(columns=[c for c in ['Loan_ID', 'Loan_Status'] if c in test.columns])
    test_pred = final.predict(test_X)
    test_proba = final.predict_proba(test_X)[:, 1]

    sub = pd.DataFrame({
        'Loan_ID': test['Loan_ID'],
        'Loan_Status': np.where(test_pred == 1, 'Y', 'N'),
        'Probability_Approved': np.round(test_proba, 4),
    })
    sub.to_csv(os.path.join(PROC, 'submission.csv'), index=False)
    print(f"\nSubmission saved: {os.path.join(PROC, 'submission.csv')}")
    print(f"Predicted approved: {(sub['Loan_Status'] == 'Y').sum()} / {len(sub)} "
          f"({(sub['Loan_Status'] == 'Y').mean():.1%})")


if __name__ == '__main__':
    main()