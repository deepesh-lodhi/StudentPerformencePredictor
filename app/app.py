import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #000000;}
    .stApp { background: #ffffff;}
    .metric-card {
        background: Black; border-radius: 14px; padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 12px;
        border-left: 5px solid #5c67f2;
    }
    .metric-card h2 { font-size: 2rem; font-weight: 700; margin: 0; color: #1a1a2e; }
    .metric-card p  { color: #888; font-size: 0.85rem; margin: 4px 0 0; }
    .result-pass { background: #e8faf0; border: 2px solid #27ae60; border-radius: 12px; padding: 18px; text-align: center; }
    .result-fail { background: #fdecea; border: 2px solid #e74c3c; border-radius: 12px; padding: 18px; text-align: center; }
    .grade-badge {
        display: inline-block; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 1.1rem;
    }
    .risk-high   { background: #fdecea; color: #c0392b; border: 1px solid #e74c3c; }
    .risk-medium { background: #fef9e7; color: #d68910; border: 1px solid #f39c12; }
    .risk-low    { background: #e8faf0; color: #1e8449; border: 1px solid #27ae60; }
    .sidebar-header {
        background: linear-gradient(135deg, #5c67f2, #7b5cf0);
        color: Black; padding: 18px 16px; border-radius: 12px;
        margin-bottom: 20px; text-align: center;
    }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .stButton > button {
        width: 100%; background: linear-gradient(135deg, #5c67f2, #7b5cf0);
        color: Black; border: none; border-radius: 10px; padding: 12px;
        font-size: 1rem; font-weight: 600; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; color: Black; }
    h1 { color: #1a1a2e; }
    h2 { color: #2c3e6b; }
    h3 { color: #34495e; }

    /* ============================================================
       FORCE LIGHT THEME EVERYWHERE
       This overrides Streamlit's automatic dark mode (which happens
       when the OS/browser is set to dark mode) so the app always
       looks the same: white/light background + dark, readable text.
       ============================================================ */

    /* Force light backgrounds on every container Streamlit might darken */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    section[data-testid="stSidebar"], .main, .block-container {
        background-color: #f7f8fc !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }

    /* Force dark, readable text everywhere by default */
    html, body, .stApp, .main, .block-container,
    p, span, label, li, div {
        color: #1a1a2e !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 { color: #1a1a2e !important; }

    /* Widget labels (sliders, selectboxes, text inputs, radio, checkbox, number input) */
    .stSlider label, .stSelectbox label, .stTextInput label,
    .stNumberInput label, .stRadio label, .stCheckbox label,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {
        color: #1a1a2e !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Slider min/max tick labels and current value */
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"],
    div[data-testid="stThumbValue"] {
        color: #1a1a2e !important;
    }

    /* Radio / selectbox option text */
    div[role="radiogroup"] label, div[data-baseweb="select"] * {
        color: #1a1a2e !important;
    }

    /* Dataframe / table text */
    [data-testid="stDataFrame"] * { color: #1a1a2e !important; }

    /* Keep elements that are SUPPOSED to be white-on-color (buttons, sidebar header, result cards) readable */
    .stButton > button, .stButton > button * { color: #ffffff !important; }
    .sidebar-header, .sidebar-header * { color: #ffffff !important; }
    .result-pass, .result-pass * { color: #1a1a2e !important; }
    .result-fail, .result-fail * { color: #1a1a2e !important; }

    /* Metric widgets (st.metric) */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricDelta"] {
        color: #1a1a2e !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import os
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models') + os.sep
    with open(base+'scaler.pkl',   'rb') as f: scaler = pickle.load(f)
    with open(base+'rf_reg.pkl',   'rb') as f: rf_reg = pickle.load(f)
    with open(base+'lin_reg.pkl',  'rb') as f: lin_reg = pickle.load(f)
    with open(base+'rf_clf.pkl',   'rb') as f: rf_clf = pickle.load(f)
    with open(base+'log_reg.pkl',  'rb') as f: log_reg = pickle.load(f)
    with open(base+'svm.pkl',      'rb') as f: svm = pickle.load(f)
    with open(base+'gb_reg.pkl',   'rb') as f: gb_reg = pickle.load(f)
    try:
      with open(base+'feature_importance.pkl','rb') as f: fi = pickle.load(f)
    except Exception:
      fi = None
    with open(base+'confusion_matrix.pkl',  'rb') as f: cm = pickle.load(f)
    with open(base+'clf_report.pkl',        'rb') as f: cr = pickle.load(f)
    return scaler, rf_reg, lin_reg, rf_clf, log_reg, svm, gb_reg, fi, cm, cr

@st.cache_data
def load_data():
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, 'data', 'student_data.csv'))

scaler, rf_reg, lin_reg, rf_clf, log_reg, svm_m, gb_reg, feat_imp, conf_mat, clf_rep = load_models()
df = load_data()

FEATURES = ['attendance','prev_marks','study_hours','assignments',
            'internals','extracurricular','parental_edu','internet',
            'health','absences']

# ── Helpers ───────────────────────────────────────────────────────────────────
GRADE_COLOR = {'O':'#1abc9c','A+':'#27ae60','A':'#2980b9','B+':'#8e44ad',
               'B':'#f39c12','C':'#e67e22','F':'#e74c3c'}

def score_to_grade(s):
    if s>=90: return 'O'
    elif s>=80: return 'A+'
    elif s>=70: return 'A'
    elif s>=60: return 'B+'
    elif s>=50: return 'B'
    elif s>=40: return 'C'
    else: return 'F'

def get_risk(score):
    if score < 45: return 'HIGH', '🔴'
    elif score < 60: return 'MEDIUM', '🟡'
    else: return 'LOW', '🟢'

def predict_all(row_dict):
    arr = np.array([[row_dict[f] for f in FEATURES]])
    arr_s = scaler.transform(arr)
    score_rf  = float(np.clip(rf_reg.predict(arr_s)[0], 0, 100))
    score_lr  = float(np.clip(lin_reg.predict(arr_s)[0], 0, 100))
    score_gb  = float(np.clip(gb_reg.predict(arr_s)[0], 0, 100))
    avg_score = round((score_rf + score_lr + score_gb) / 3, 1)
    grade     = score_to_grade(avg_score)
    status    = 'PASS' if avg_score >= 40 else 'FAIL'
    risk, risk_icon = get_risk(avg_score)
    prob_pass = float(rf_clf.predict_proba(arr_s)[0][1]) * 100
    return {
        'score_rf': round(score_rf,1), 'score_lr': round(score_lr,1),
        'score_gb': round(score_gb,1), 'avg_score': avg_score,
        'grade': grade, 'status': status,
        'risk': risk, 'risk_icon': risk_icon, 'prob_pass': round(prob_pass,1)
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size:2rem">🎓</div>
        <div style="font-weight:700;font-size:1.1rem">SPP System</div>
        <div style="font-size:0.78rem;opacity:0.85">Student Performance Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Dashboard",
        "🔮  Predict",
        "👥  Batch Analysis",
        "📊  EDA & Analytics",
        "🤖  Model Performance",
        "ℹ️  About"
    ])
    st.markdown("---")
    st.markdown("<small style='color:#aaa'>RGPV Bhopal • CSE (Data Science)<br>Minor Project II • 2025</small>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.title("📊 Dashboard — Overview")
    st.markdown("Academic performance summary for the current batch.")
    st.markdown("---")

    total    = len(df)
    avg_sc   = df['final_score'].mean()
    pass_pct = (df['result'] == 'PASS').mean() * 100
    at_risk  = (df['final_score'] < 50).sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👨‍🎓 Total Students", total)
    c2.metric("📈 Avg Score",       f"{avg_sc:.1f}%")
    c3.metric("✅ Pass Rate",        f"{pass_pct:.1f}%")
    c4.metric("⚠️ At Risk (<50%)",   at_risk)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grade Distribution")
        grade_counts = df['grade'].value_counts().reindex(['O','A+','A','B+','B','C','F'], fill_value=0)
        fig, ax = plt.subplots(figsize=(6,4))
        colors = [GRADE_COLOR[g] for g in grade_counts.index]
        bars = ax.bar(grade_counts.index, grade_counts.values, color=colors, edgecolor='Black', linewidth=1.5, width=0.6)
        for bar, val in zip(bars, grade_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(val), ha='center', fontweight='bold', fontsize=10)
        ax.set_xlabel("Grade", fontsize=11); ax.set_ylabel("Number of Students", fontsize=11)
        ax.set_facecolor("#150101"); fig.patch.set_facecolor('Black')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df['final_score'], bins=25, color='#5c67f2', edgecolor='white', linewidth=0.8, alpha=0.85)
        ax.axvline(df['final_score'].mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean: {avg_sc:.1f}")
        ax.axvline(40, color='#f39c12', linestyle=':', linewidth=2, label="Pass Line (40)")
        ax.set_xlabel("Final Score", fontsize=11); ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_facecolor("#0e0202"); fig.patch.set_facecolor('Black')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Score by Attendance Bracket")
        df['att_bracket'] = pd.cut(df['attendance'], bins=[0,50,60,70,80,90,100],
                                   labels=['<50','50-60','60-70','70-80','80-90','90+'])
        grp = df.groupby('att_bracket', observed=True)['final_score'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(grp['att_bracket'].astype(str), grp['final_score'],
               color="#09130d", alpha=0.85, edgecolor='Black', linewidth=1.2, width=0.6)
        ax.set_xlabel("Attendance %", fontsize=11); ax.set_ylabel("Avg Final Score", fontsize=11)
        ax.set_facecolor("#150d0d"); fig.patch.set_facecolor('Black')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Study Hours vs Final Score")
        fig, ax = plt.subplots(figsize=(6,4))
        sc = ax.scatter(df['study_hours'], df['final_score'],
                        c=df['final_score'], cmap='RdYlGn', alpha=0.6, edgecolors='none', s=25)
        plt.colorbar(sc, ax=ax, label='Score')
        z = np.polyfit(df['study_hours'], df['final_score'], 1)
        p = np.poly1d(z)
        xs = np.linspace(df['study_hours'].min(), df['study_hours'].max(), 100)
        ax.plot(xs, p(xs), 'r--', linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Daily Study Hours", fontsize=11); ax.set_ylabel("Final Score", fontsize=11)
        ax.set_facecolor("#0A0202"); fig.patch.set_facecolor('Black')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict":
    st.title("🔮 Predict Student Performance")
    st.markdown("Enter student academic parameters below. The system uses 3 ML models and returns an ensemble prediction.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("📋 Student Parameters")
        name        = st.text_input("Student Name (optional)", placeholder="e.g. Arjun Sharma")
        attendance  = st.slider("📅 Attendance %",         0,  100, 75)
        prev_marks  = st.slider("📝 Previous Sem Marks %", 0,  100, 65)
        internals   = st.slider("📋 Internal Marks %",     0,  100, 60)
        assignments = st.slider("📁 Assignment Completion %", 0, 100, 70)
        study_hours = st.slider("⏰ Daily Study Hours",    0.0, 12.0, 4.0, 0.5)
        absences    = st.slider("🚫 Absences (days)",      0,  30,   4)
        health      = st.slider("❤️ Health (1=Poor, 5=Excellent)", 1, 5, 3)

        c1, c2 = st.columns(2)
        with c1:
            extracurricular = st.selectbox("🎭 Extracurricular", ["No", "Yes"])
            parental_edu    = st.selectbox("👨‍👩‍👧 Parental Education", ["None","Graduate","Postgrad"])
        with c2:
            internet = st.selectbox("🌐 Internet Access", ["No", "Yes"])

        st.markdown("")
        predict_btn = st.button("🚀 Run Prediction")

    with col_result:
        st.subheader("📊 Prediction Result")
        if predict_btn:
            row = {
                'attendance':   attendance,
                'prev_marks':   prev_marks,
                'study_hours':  study_hours,
                'assignments':  assignments,
                'internals':    internals,
                'extracurricular': 1 if extracurricular=="Yes" else 0,
                'parental_edu': ["None","Graduate","Postgrad"].index(parental_edu),
                'internet':     1 if internet=="Yes" else 0,
                'health':       health,
                'absences':     absences,
            }
            with st.spinner("Running ensemble model..."):
                res = predict_all(row)

            # Result card
            status_class = "result-pass" if res['status']=="PASS" else "result-fail"
            status_emoji = "✅" if res['status']=="PASS" else "❌"
            st.markdown(f"""
            <div class="{status_class}">
                <h1 style="margin:0">{status_emoji} {res['status']}</h1>
                <h2 style="margin:8px 0 0">Score: {res['avg_score']}%</h2>
                <p style="margin:4px 0 0; font-size:0.9rem">Grade: <strong>{res['grade']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            r1, r2 = st.columns(2)
            risk_cls = f"risk-{res['risk'].lower()}"
            r1.markdown(f"""
            <div class="metric-card" style="border-color:{'#e74c3c' if res['risk']=='HIGH' else '#f39c12' if res['risk']=='MEDIUM' else '#27ae60'}">
                <p>RISK LEVEL</p>
                <h2>{res['risk_icon']} {res['risk']}</h2>
            </div>""", unsafe_allow_html=True)
            r2.markdown(f"""
            <div class="metric-card" style="border-color:#5c67f2">
                <p>PASS PROBABILITY</p>
                <h2>{res['prob_pass']}%</h2>
            </div>""", unsafe_allow_html=True)

            # Model breakdown
            st.subheader("Model Scores")
            model_df = pd.DataFrame({
                'Model':['Random Forest','Linear Reg','Gradient Boost','Ensemble'],
                'Score':[res['score_rf'],res['score_lr'],res['score_gb'],res['avg_score']]
            })
            fig, ax = plt.subplots(figsize=(5,2.5))
            colors_m = ['#5c67f2','#27ae60','#e67e22','#e74c3c']
            bars = ax.barh(model_df['Model'], model_df['Score'],
                           color=colors_m, edgecolor='white', linewidth=1, height=0.5)
            for bar, val in zip(bars, model_df['Score']):
                ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}", va='center', fontweight='bold', fontsize=10)
            ax.set_xlim(0, 105); ax.set_xlabel("Score (%)")
            ax.axvline(40, color='red', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_facecolor("#110202"); fig.patch.set_facecolor('Black')
            ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig); plt.close()

            # Suggestions
            if res['risk'] != 'LOW':
                st.warning("⚠️ **Improvement Suggestions**")
                tips = []
                if attendance < 75:  tips.append("• Improve attendance (target ≥ 75%)")
                if study_hours < 4:  tips.append("• Increase daily study time (target ≥ 4 hrs)")
                if assignments < 70: tips.append("• Submit more assignments (target ≥ 70%)")
                if absences > 5:     tips.append("• Reduce absences (target < 5 days)")
                if internals < 50:   tips.append("• Focus on internal exam preparation")
                for t in tips: st.markdown(t)
            else:
                st.success("✅ Keep up the great work! You are on track.")

        else:
            st.info("👈 Fill in the parameters on the left and click **Run Prediction**.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥  Batch Analysis":
    st.title("👥 Batch Analysis")
    st.markdown("Upload a CSV file with student data to get predictions for the entire class.")

    template_df = pd.DataFrame([{
        'attendance':85,'prev_marks':72,'study_hours':5,'assignments':80,
        'internals':65,'extracurricular':1,'parental_edu':1,
        'internet':1,'health':3,'absences':3
    }])

    st.download_button("⬇️ Download Template CSV", template_df.to_csv(index=False),
                       "student_template.csv", "text/csv")
    st.markdown("---")

    uploaded = st.file_uploader("Upload Student Data CSV", type=['csv'])

    if uploaded:
        udf = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in udf.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.success(f"✅ Loaded {len(udf)} student records")
            results_list = []
            for _, row in udf.iterrows():
                r = predict_all(row.to_dict())
                results_list.append(r)
            res_df = udf.copy()
            res_df['Predicted Score'] = [r['avg_score'] for r in results_list]
            res_df['Grade']           = [r['grade']     for r in results_list]
            res_df['Status']          = [r['status']    for r in results_list]
            res_df['Risk Level']      = [r['risk']      for r in results_list]
            res_df['Pass Prob %']     = [r['prob_pass'] for r in results_list]

            st.subheader("Results")
            st.dataframe(res_df[['Predicted Score','Grade','Status','Risk Level','Pass Prob %']])

            c1,c2,c3 = st.columns(3)
            c1.metric("Total Students", len(res_df))
            c2.metric("Pass Count",     (res_df['Status']=='PASS').sum())
            c3.metric("At Risk (HIGH)", (res_df['Risk Level']=='HIGH').sum())

            st.download_button("⬇️ Download Results CSV",
                               res_df.to_csv(index=False), "predictions.csv","text/csv")
    else:
        st.markdown("### 📋 Sample Data Preview (Training Dataset)")
        sample = df.sample(10, random_state=1)[FEATURES + ['final_score','grade','result']].reset_index(drop=True)
        st.dataframe(sample)

        # Predict on sample
        st.subheader("Sample Predictions")
        preds = []
        for _, row in sample.iterrows():
            r = predict_all(row.to_dict())
            preds.append({'Predicted Score': r['avg_score'], 'Grade': r['grade'],
                          'Status': r['status'], 'Risk': r['risk'], 'Pass Prob %': r['prob_pass']})
        pred_df = pd.DataFrame(preds)
        combined = pd.concat([sample[FEATURES].reset_index(drop=True), pred_df], axis=1)
        st.dataframe(combined[['attendance','prev_marks','study_hours','Predicted Score','Grade','Status','Risk','Pass Prob %']])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA & Analytics":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Deep-dive into the dataset to understand patterns and relationships.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📉 Feature Analysis"])

    with tab1:
        st.subheader("Feature Distributions")
        num_features = ['attendance','prev_marks','study_hours','assignments','internals','absences']
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        colors_list = ['#5c67f2','#27ae60','#e67e22','#8e44ad','#2980b9','#e74c3c']
        for ax, feat, col in zip(axes.flat, num_features, colors_list):
            ax.hist(df[feat], bins=20, color=col, edgecolor='white', linewidth=0.8, alpha=0.85)
            ax.axvline(df[feat].mean(), color='black', linestyle='--', linewidth=1.5, label=f"μ={df[feat].mean():.1f}")
            ax.set_title(feat.replace('_',' ').title(), fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_facecolor("#0e0303")
            ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('Black')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Correlation Heatmap")
        corr = df[FEATURES + ['final_score']].corr()
        fig, ax = plt.subplots(figsize=(10,7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, linewidths=0.5, cbar_kws={'shrink':0.8})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Feature Importance (Random Forest)")
        fi_sorted = feat_imp.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi_sorted)))
        bars = ax.barh(fi_sorted.index, fi_sorted.values, color=colors_fi, edgecolor='Black', height=0.6)
        for bar, val in zip(bars, fi_sorted.values):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel("Importance Score"); ax.set_title("Feature Importance", fontweight='bold')
        ax.set_facecolor("#0d0101"); fig.patch.set_facecolor('Black')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Feature vs Final Score")
        feat_choice = st.selectbox("Select Feature", ['attendance','prev_marks','study_hours','absences','internals'])
        fig, ax = plt.subplots(figsize=(9, 4))
        sc = ax.scatter(df[feat_choice], df['final_score'],
                        c=df['final_score'], cmap='RdYlGn', alpha=0.5, s=20, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Final Score')
        z = np.polyfit(df[feat_choice], df['final_score'], 1)
        p = np.poly1d(z)
        xs = np.linspace(df[feat_choice].min(), df[feat_choice].max(), 100)
        ax.plot(xs, p(xs), 'b--', linewidth=2, label='Trend Line')
        ax.set_xlabel(feat_choice.replace('_',' ').title(), fontsize=11)
        ax.set_ylabel("Final Score", fontsize=11)
        corr_val = df[feat_choice].corr(df['final_score'])
        ax.set_title(f"Pearson Correlation: r = {corr_val:.3f}", fontweight='bold')
        ax.legend(); ax.set_facecolor("#0c0404"); fig.patch.set_facecolor('Black')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Boxplot: Final Score by Grade")
        grade_order = ['O','A+','A','B+','B','C','F']
        fig, ax = plt.subplots(figsize=(9,4))
        data_by_grade = [df[df['grade']==g]['final_score'].values for g in grade_order if g in df['grade'].values]
        labels_by_grade = [g for g in grade_order if g in df['grade'].values]
        bp = ax.boxplot(data_by_grade, labels=labels_by_grade, patch_artist=True, notch=False,
                        medianprops={'color':'black','linewidth':2})
        for patch, g in zip(bp['boxes'], labels_by_grade):
            patch.set_facecolor(GRADE_COLOR.get(g,'#5c67f2'))
            patch.set_alpha(0.75)
        ax.set_xlabel("Grade", fontsize=11); ax.set_ylabel("Final Score", fontsize=11)
        ax.set_facecolor("#0c0202"); fig.patch.set_facecolor('Black')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.title("🤖 Model Performance & Evaluation")
    st.markdown("Detailed evaluation of all trained ML models.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Regression Models", "🎯 Classification Models", "🔬 Confusion Matrix"])

    with tab1:
        st.subheader("Regression Models — Score Prediction")
        reg_data = {
            'Model':       ['Linear Regression','Random Forest Reg','Gradient Boosting'],
            'RMSE':        [6.29, 7.61, 7.15],
            'R² Score':    [0.7646, 0.6557, 0.6957],
            'Training Time':['Fast','Medium','Slow']
        }
        st.dataframe(pd.DataFrame(reg_data), use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        models = ['Linear\nRegression','Random\nForest','Gradient\nBoosting']
        rmse_vals = [6.29, 7.61, 7.15]
        r2_vals   = [0.7646, 0.6557, 0.6957]
        colors_b  = ['#5c67f2','#27ae60','#e67e22']

        axes[0].bar(models, rmse_vals, color=colors_b, edgecolor='Black', linewidth=1.5, width=0.5)
        axes[0].set_title("RMSE Comparison (Lower = Better)", fontweight='bold')
        axes[0].set_ylabel("RMSE"); axes[0].set_facecolor("#0c0202")
        axes[0].spines[['top','right']].set_visible(False)
        for bar, val in zip(axes[0].patches, rmse_vals):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                         f"{val}", ha='center', fontweight='bold')

        axes[1].bar(models, r2_vals, color=colors_b, edgecolor='white', linewidth=1.5, width=0.5)
        axes[1].set_title("R² Score Comparison (Higher = Better)", fontweight='bold')
        axes[1].set_ylabel("R² Score"); axes[1].set_ylim(0, 1)
        axes[1].set_facecolor("#0e0303")
        axes[1].spines[['top','right']].set_visible(False)
        for bar, val in zip(axes[1].patches, r2_vals):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                         f"{val:.4f}", ha='center', fontweight='bold', fontsize=9)

        fig.patch.set_facecolor('Black')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Classification Models — Pass/Fail Prediction")
        clf_data = {
            'Model':    ['Logistic Regression','Random Forest Clf','SVM (RBF)'],
            'Accuracy': ['93.75%', '93.75%', '93.75%'],
            'Notes':    ['Interpretable, fast','Robust, handles non-linearity','Good with high-dim data']
        }
        st.dataframe(pd.DataFrame(clf_data), use_container_width=True)

        st.subheader("Classification Report (Random Forest)")
        metrics = {
            'Class':     ['FAIL', 'PASS'],
            'Precision': [round(clf_rep.get('0',{}).get('precision',0),2),
                          round(clf_rep.get('1',{}).get('precision',0),2)],
            'Recall':    [round(clf_rep.get('0',{}).get('recall',0),2),
                          round(clf_rep.get('1',{}).get('recall',0),2)],
            'F1-Score':  [round(clf_rep.get('0',{}).get('f1-score',0),2),
                          round(clf_rep.get('1',{}).get('f1-score',0),2)],
            'Support':   [int(clf_rep.get('0',{}).get('support',0)),
                          int(clf_rep.get('1',{}).get('support',0))],
        }
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)

    with tab3:
        st.subheader("Confusion Matrix — Random Forest Classifier")
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['FAIL','PASS'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title("Confusion Matrix (RF Classifier)", fontweight='bold', fontsize=13)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        tn, fp, fn, tp = conf_mat.ravel()
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("True Positive",  int(tp), help="Correctly predicted PASS")
        col2.metric("True Negative",  int(tn), help="Correctly predicted FAIL")
        col3.metric("False Positive", int(fp), help="FAIL predicted as PASS")
        col4.metric("False Negative", int(fn), help="PASS predicted as FAIL")

        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall    = tp/(tp+fn) if (tp+fn)>0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        c1,c2,c3 = st.columns(3)
        c1.metric("Precision", f"{precision:.3f}")
        c2.metric("Recall",    f"{recall:.3f}")
        c3.metric("F1 Score",  f"{f1:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎓 Project Details")
        details = {
            "Project Title":  "Student Performance Predictor",
            "Project Type":   "Minor Project II",
            "University":     "RGPV Bhopal",
            "School":         "School of Information Technology",
            "Branch":         "CSE (Data Science)",
            "Semester":       "6th Semester",
            "Session":        "2024–25",
        }
        
    with col2:
        st.subheader("⚙️ Tech Stack")
        tech = ["Python 3.10", "Scikit-learn", "Pandas", "NumPy",
                "Matplotlib", "Seaborn", "Streamlit", "Pickle"]
        cols = st.columns(2)
        for i, t in enumerate(tech):
            cols[i%2].markdown(f"✅ `{t}`")

        st.markdown("---")
        st.subheader("🎯 Objectives")
        objectives = [
            "Analyze student academic data to identify key performance indicators",
            "Train ML models (LR, RF, SVM, GB) for score and pass/fail prediction",
            "Build interactive Streamlit dashboard with real-time predictions",
            "Evaluate models using RMSE, R², accuracy, precision, recall, F1",
        ]
        for i, obj in enumerate(objectives, 1):
            st.markdown(f"**{i}.** {obj}")

        st.markdown("---")
        st.subheader("📦 Dataset Info")
        st.markdown(f"""
        - **Records:** {len(df)} synthetic student entries
        - **Features:** 10 academic & behavioral attributes
        - **Target:** Final score (regression) + Pass/Fail (classification)
        - **Grade Scale:** O / A+ / A / B+ / B / C / F
        """)

    #st.markdown("---")
    #st.info("🏫 **School of Information Technology** | Rajiv Gandhi Proudyogiki Vishwavidyalaya, Bhopal")
