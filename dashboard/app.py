"""Loan Approval Prediction Dashboard — styled interface.

Streamlit UI that loads the trained model and predicts whether a loan
applicant will be approved. Supports single-applicant and batch prediction.

Run from the project root:
    python -m streamlit run dashboard/app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- paths
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE, 'models', 'final_model.pkl')

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- custom CSS
st.markdown("""
<style>
:root {
    --page: #080d27;
    --panel: #121b3d;
    --panel-raised: #18244b;
    --border: #25315b;
    --text: #f2f4ff;
    --muted: #9aa8ce;
    --blue: #169bff;
    --purple: #7618f2;
    --green: #08d899;
}

.stApp {
    background: var(--page);
    color: var(--text);
}
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b9b 0%, #11156f 100%);
    border-right: 1px solid #29368f;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}
[data-testid="stSidebar"] * {
    color: #e8edff;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #aab8ff;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: .45rem;
}
[data-testid="stSidebar"] [role="radio"] {
    padding: .68rem .75rem;
    border: 1px solid transparent;
    border-radius: 9px;
}
[data-testid="stSidebar"] [role="radio"]:has(input:checked) {
    background: rgba(5, 12, 57, .55);
    border-color: rgba(148, 166, 255, .28);
    color: #fff;
    font-weight: 700;
}

.hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.15rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: linear-gradient(110deg, #121b3d 0%, #111d43 72%, #17235a 100%);
    box-shadow: 0 12px 32px rgba(0, 0, 0, .18);
}
.hero h1 {
    margin: 0;
    color: var(--text);
    font-size: clamp(1.55rem, 3vw, 2rem);
    font-weight: 700;
    letter-spacing: -.04em;
}
.hero p {
    margin: 0.4rem 0 0;
    color: var(--muted);
    font-size: .92rem;
}
.hero-status {
    flex-shrink: 0;
    padding: .55rem .8rem;
    border: 1px solid rgba(8, 216, 153, .28);
    border-radius: 999px;
    background: rgba(8, 216, 153, .09);
    color: #70f0c4;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .06em;
}

.metric-card {
    min-height: 100px;
    padding: 1rem 1.1rem;
    margin-bottom: .8rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
}
.metric-card h4 {
    margin: 0 0 .45rem;
    color: var(--muted);
    font-size: .75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.metric-card .value {
    color: var(--text);
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -.03em;
}

.dashboard-card {
    min-height: 250px;
    padding: 1.25rem 1.35rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
}
.dashboard-card h3 {
    margin: 0 0 .25rem;
    color: var(--text);
    font-size: 1.05rem;
}
.dashboard-card .card-subtitle {
    margin: 0;
    color: var(--muted);
    font-size: .8rem;
}
.outcome-chart {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    min-height: 180px;
}
.donut {
    display: grid;
    width: 148px;
    height: 148px;
    flex: 0 0 148px;
    place-items: center;
    border-radius: 50%;
    background: conic-gradient(var(--blue) 0 69.1%, var(--purple) 69.1% 100%);
}
.donut-hole {
    display: flex;
    width: 96px;
    height: 96px;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: var(--panel);
    color: var(--text);
    font-size: 1.35rem;
    font-weight: 700;
}
.donut-hole span {
    color: var(--muted);
    font-size: .65rem;
    font-weight: 500;
}
.legend {
    display: grid;
    gap: .8rem;
    color: var(--muted);
    font-size: .82rem;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: .55rem;
}
.legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 3px;
}
.performance-list {
    display: grid;
    gap: 1.05rem;
    margin-top: 1.35rem;
}
.performance-row {
    display: grid;
    grid-template-columns: 90px 1fr 48px;
    align-items: center;
    gap: .65rem;
    color: var(--muted);
    font-size: .78rem;
}
.performance-track {
    height: 8px;
    overflow: hidden;
    border-radius: 10px;
    background: #273257;
}
.performance-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--blue), var(--purple));
}
.performance-value {
    color: var(--text);
    text-align: right;
    font-weight: 700;
}

.result-approved {
    padding: 1.35rem 1.5rem;
    margin-top: .5rem;
    border: 1px solid rgba(8, 216, 153, .35);
    border-radius: 12px;
    background: linear-gradient(110deg, #102d3a, #123b3b);
    color: var(--text);
}
.result-rejected {
    padding: 1.35rem 1.5rem;
    margin-top: .5rem;
    border: 1px solid rgba(255, 99, 132, .35);
    border-radius: 12px;
    background: linear-gradient(110deg, #2d1d39, #3a1c36);
    color: var(--text);
}
.result-approved h2, .result-rejected h2 {
    margin: 0 0 .3rem;
    color: var(--text);
    font-size: 1.45rem;
    font-weight: 700;
}
.result-approved p, .result-rejected p {
    margin: 0;
    color: #d3d9ef;
    font-size: 1rem;
}

div[data-testid="stForm"] {
    padding: 1.2rem 1.35rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
}
[data-testid="stWidgetLabel"] p {
    color: #cbd4f0;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-color: #35436d;
    border-radius: 8px;
    background: #0c1432;
}
div[data-baseweb="select"] *,
div[data-baseweb="input"] * {
    color: var(--text);
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    min-height: 2.7rem;
    border: 0;
    border-radius: 8px;
    background: linear-gradient(105deg, var(--blue), var(--purple));
    color: #fff;
    font-weight: 700;
    box-shadow: 0 5px 18px rgba(67, 87, 255, .25);
    transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    border: 0;
    color: #fff;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(67, 87, 255, .38);
}
div.stButton > button:focus-visible,
div[data-testid="stFormSubmitButton"] > button:focus-visible,
div[data-testid="stDownloadButton"] > button:focus-visible {
    outline: 2px solid #9db8ff;
    outline-offset: 2px;
}
div[data-testid="stDownloadButton"] > button {
    min-height: 2.7rem;
    border: 1px solid rgba(8, 216, 153, .35);
    border-radius: 8px;
    background: #123b3b;
    color: #8bf0cb;
    font-weight: 700;
}
[data-testid="stFileUploader"] section {
    border-color: #35436d;
    border-radius: 9px;
    background: var(--panel);
}
[data-testid="stFileUploader"] button {
    border: 1px solid #169bff !important;
    border-radius: 7px !important;
    background: #147fd1 !important;
    color: #fff !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] button * {
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: inherit !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: #45afff !important;
    background: #096ab8 !important;
    color: #fff !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploader"] small {
    color: #b7c8ff !important;
}
[data-testid="stFileUploader"] button:focus-visible {
    outline: 2px solid #9db8ff;
    outline-offset: 2px;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 9px;
}
[data-testid="stAlert"] {
    border-radius: 9px;
    background: var(--panel-raised);
    color: var(--text);
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--blue), var(--purple));
}
.stMarkdown p, [data-testid="stCaptionContainer"] {
    color: var(--muted);
}
[data-testid="stMarkdownContainer"] code {
    padding: .12em .35em;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: #0c1432 !important;
    color: #b7c8ff !important;
}
hr {
    border-color: var(--border);
}
.footer-note {
    text-align: center;
    color: #7f8db6;
    font-size: .78rem;
    padding: 1.2rem 0 0.5rem;
}
@media (max-width: 700px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }
    .hero {
        align-items: flex-start;
        flex-direction: column;
        padding: 1.2rem;
    }
    .outcome-chart {
        flex-direction: column;
        padding: 1rem 0;
    }
    .dashboard-card {
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL):
        st.error(f"Model not found at {MODEL}. Run `python src/train_models.py` first.")
        st.stop()
    return joblib.load(MODEL)

model = load_model()

# ---------------------------------------------------------------- helpers
MODEL_COLS = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area',
    'ApplicantIncome_log', 'CoapplicantIncome_log', 'LoanAmount_log',
    'LoanAmount_capped',
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ApplicantIncome_log']   = np.log1p(df['ApplicantIncome'].fillna(0))
    df['CoapplicantIncome_log'] = np.log1p(df['CoapplicantIncome'].fillna(0))
    df['LoanAmount_log']        = np.log1p(df['LoanAmount'].fillna(0))
    df['LoanAmount_capped']     = df['LoanAmount'].clip(upper=496.36)
    df['Dependents'] = (df['Dependents']
                        .astype(str).str.replace('3+', '3', regex=False)
                        .replace('nan', np.nan).astype(float))
    return df

def predict(df: pd.DataFrame):
    df_eng = engineer_features(df)
    X = df_eng[MODEL_COLS]
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return pred, proba

# ---------------------------------------------------------------- navigation
with st.sidebar:
    st.markdown("## LoanFlow")
    st.caption("LOAN MANAGEMENT")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Overview", "Single Applicant", "Batch Upload"],
        label_visibility="visible",
    )
    st.divider()
    st.caption("MODEL STATUS")
    st.markdown(" Random Forest · Ready")
    st.caption("DIAL3110 · Motz Financial Services")

if page == "Overview":
    st.markdown("""
    <div class="hero">
        <div>
            <h1>Loan portfolio overview</h1>
            <p>Model performance and approval outcomes at a glance.</p>
        </div>
        <div class="hero-status">● MODEL ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for column, label, value in zip(
        [c1, c2, c3, c4],
        ["Model", "Holdout accuracy", "ROC-AUC", "Test applications"],
        ["Random Forest", "86.18%", "0.8053", "123"],
    ):
        with column:
            st.markdown(
                f'<div class="metric-card"><h4>{label}</h4>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.markdown("""
        <div class="dashboard-card">
            <h3>Holdout outcomes</h3>
            <p class="card-subtitle">Prediction results across 123 test applications</p>
            <div class="outcome-chart">
                <div class="donut">
                    <div class="donut-hole">123<span>TEST CASES</span></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <span class="legend-dot" style="background:#169bff"></span>
                        Approved <strong style="color:#f2f4ff">85</strong>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background:#7618f2"></span>
                        Not approved <strong style="color:#f2f4ff">38</strong>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="dashboard-card">
            <h3>Model performance</h3>
            <p class="card-subtitle">Evaluation metrics on the held-out test set</p>
            <div class="performance-list">
                <div class="performance-row">
                    <span>Accuracy</span>
                    <div class="performance-track"><div class="performance-fill" style="width:86.18%"></div></div>
                    <span class="performance-value">86.18%</span>
                </div>
                <div class="performance-row">
                    <span>ROC-AUC</span>
                    <div class="performance-track"><div class="performance-fill" style="width:80.53%"></div></div>
                    <span class="performance-value">80.53%</span>
                </div>
                <div class="performance-row">
                    <span>Precision</span>
                    <div class="performance-track"><div class="performance-fill" style="width:85.42%"></div></div>
                    <span class="performance-value">85.42%</span>
                </div>
                <div class="performance-row">
                    <span>Recall</span>
                    <div class="performance-track"><div class="performance-fill" style="width:96.47%"></div></div>
                    <span class="performance-value">96.47%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div class="dashboard-card" style="min-height:auto">
        <h3>Prediction workspace</h3>
        <p class="card-subtitle">
            Use <strong style="color:#f2f4ff">Single Applicant</strong> to evaluate one application,
            or <strong style="color:#f2f4ff">Batch Upload</strong> to process a CSV portfolio.
            Choose a workspace from the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================ SINGLE
elif page == "Single Applicant":
    st.markdown("""
    <div class="hero">
        <div>
            <h1>Single applicant</h1>
            <p>Enter an applicant profile to estimate loan approval probability.</p>
        </div>
        <div class="hero-status">● MODEL ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)
    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Personal Details")
            gender        = st.selectbox("Gender", ["Male", "Female"])
            married       = st.selectbox("Married", ["Yes", "No"])
            dependents    = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        with col2:
            st.markdown("##### 💰 Income & Loan")
            applicant_income   = st.number_input("Applicant Income (monthly)",
                                                0, 1_000_000, 5000, step=500)
            coapplicant_income = st.number_input("Coapplicant Income (monthly)",
                                                0, 1_000_000, 0, step=500)
            loan_amount        = st.number_input("Loan Amount (thousands)",
                                                0, 10_000, 150, step=10)
            loan_term          = st.selectbox("Loan Term (months)",
                                              [360, 180, 240, 120, 84, 60, 36, 12])

        with col3:
            st.markdown("##### 📊 Credit & Property")
            credit_history = st.selectbox(
                "Credit History", ["1 — Good", "0 — Bad"])
            property_area  = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔍 Predict Loan Approval",
                                          use_container_width=True)

    if submitted:
        credit_val = int(credit_history.split(" — ")[0])
        applicant = pd.DataFrame([{
            'Gender': gender, 'Married': married, 'Dependents': dependents,
            'Education': education, 'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_term,
            'Credit_History': credit_val, 'Property_Area': property_area,
        }])

        pred, proba = predict(applicant)
        p = float(proba[0])

        if pred[0] == 1:
            st.markdown(f"""
            <div class="result-approved">
                <h2>✅ Loan Approved</h2>
                <p>Approval probability: <b>{p:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <h2>❌ Loan Not Approved</h2>
                <p>Approval probability: <b>{p:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(p)
        st.caption("Prediction generated by a Random Forest classifier. "
                   "This is a decision-support tool, not a real credit decision.")

# ============================================================ BATCH
else:
    st.markdown("""
    <div class="hero">
        <div>
            <h1>Batch upload</h1>
            <p>Process a portfolio of applicants from a CSV file.</p>
        </div>
        <div class="hero-status">● MODEL ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### 📁 Batch Prediction")
    st.markdown(
        "Upload a CSV containing applicant records. "
        "Required columns: `Gender, Married, Dependents, Education, "
        "Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, "
        "Loan_Amount_Term, Credit_History, Property_Area`. "
        "If a `Loan_Status` column is present it is treated as ground truth "
        "and used to compute accuracy — it is never used as a model input."
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            n_in = len(raw)
            X_raw = raw.drop(columns=[c for c in ['Loan_ID', 'Loan_Status']
                                      if c in raw.columns])
            pred, proba = predict(X_raw)

            out = raw.copy()
            out['Prediction'] = np.where(pred == 1, 'Y', 'N')
            out['Probability_Approved'] = np.round(proba, 4)
            assert len(out) == n_in

            approved = int((pred == 1).sum())
            rejected = int((pred == 0).sum())

            # metric strip
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><h4>Total</h4>'
                            f'<div class="value">{n_in}</div></div>',
                            unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card" style="border-left-color:#0f9b6c">'
                            f'<h4>Approved</h4>'
                            f'<div class="value" style="color:#0f9b6c">{approved}</div></div>',
                            unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card" style="border-left-color:#c0392b">'
                            f'<h4>Not Approved</h4>'
                            f'<div class="value" style="color:#c0392b">{rejected}</div></div>',
                            unsafe_allow_html=True)

            st.write("")

            cols_first = ['Loan_ID'] if 'Loan_ID' in out.columns else []
            display_cols = cols_first + ['Prediction', 'Probability_Approved']
            st.dataframe(out[display_cols], use_container_width=True, height=420)

            st.download_button(
                "⬇️ Download all predictions (CSV)",
                out.to_csv(index=False).encode(),
                "predictions.csv",
                "text/csv",
                use_container_width=True,
            )

            if 'Loan_Status' in raw.columns:
                correct = int((raw['Loan_Status'].values == out['Prediction'].values).sum())
                st.info(f"📊 Accuracy vs provided labels: "
                        f"**{correct}/{n_in} = {correct/n_in:.2%}**")

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# ---------------------------------------------------------------- footer
st.markdown('<div class="footer-note">Random Forest · max_depth=10 · '
            'min_samples_split=10 · n_estimators=100 · DIAL3110 Group Assignment</div>',
            unsafe_allow_html=True)