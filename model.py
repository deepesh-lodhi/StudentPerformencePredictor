"""
=============================================================
  Student Performance Predictor
  ML Model Training Script
  Minor Project II — CSE (Data Science) | RGPV Bhopal
  Team: NAME-1 | NAME-2 | NAME-3
=============================================================

HOW TO RUN:
    pip install scikit-learn pandas numpy matplotlib seaborn
    python model.py

OUTPUT:
    models/  -> all trained .pkl files
    plots/   -> all saved charts
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, precision_score, recall_score, f1_score
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = 'student_data.csv'
MODEL_DIR  = 'models'
PLOT_DIR   = 'plots'
FEATURES   = ['attendance','prev_marks','study_hours','assignments',
              'internals','extracurricular','parental_edu','internet',
              'health','absences']
RANDOM_STATE = 42
TEST_SIZE    = 0.20

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ── Helper Functions ──────────────────────────────────────────────────────────
def score_to_grade(score):
    """Convert numeric score to letter grade (RGPV 10-point scale)"""
    if score >= 90: return 'O'
    elif score >= 80: return 'A+'
    elif score >= 70: return 'A'
    elif score >= 60: return 'B+'
    elif score >= 50: return 'B'
    elif score >= 40: return 'C'
    else: return 'F'

def get_risk(score):
    if score < 45:   return 'HIGH',   '🔴'
    elif score < 60: return 'MEDIUM', '🟡'
    else:            return 'LOW',    '🟢'

def save_pkl(obj, name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")

def load_pkl(name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'rb') as f:
        return pickle.load(f)

def sep(title=""):
    print("\n" + "="*60)
    if title: print(f"  {title}")
    print("="*60)

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Generate / Load Dataset
# ═══════════════════════════════════════════════════════════════════
sep("STEP 1: DATASET")

if not os.path.exists(DATA_PATH):
    print("Generating synthetic dataset...")
    np.random.seed(RANDOM_STATE)
    n = 400
    attendance  = np.clip(np.random.normal(72, 18, n), 20, 100).round(1)
    prev_marks  = np.clip(np.random.normal(60, 22, n), 10, 100).round(1)
    study_hours = np.clip(np.random.normal(3.5, 2.2, n), 0, 12).round(1)
    assignments = np.clip(np.random.normal(68, 22, n), 10, 100).round(1)
    internals   = np.clip(np.random.normal(55, 22, n), 10, 100).round(1)
    extracurricular = np.random.choice([0,1], n, p=[0.55,0.45])
    parental_edu    = np.random.choice([0,1,2], n, p=[0.3,0.4,0.3])
    internet        = np.random.choice([0,1], n, p=[0.3,0.7])
    health          = np.random.randint(1, 6, n)
    absences        = np.clip(np.random.poisson(5, n), 0, 30)
    final_score = np.clip(
        0.25*attendance + 0.30*prev_marks + 0.20*internals +
        0.10*assignments + study_hours*2.2 + extracurricular*2 +
        parental_edu*1.5 + internet*1.5 - absences*1.0 + health*1.0 +
        np.random.normal(0, 6, n), 0, 100
    ).round(1)
    df = pd.DataFrame({
        'attendance':attendance,'prev_marks':prev_marks,'study_hours':study_hours,
        'assignments':assignments,'internals':internals,'extracurricular':extracurricular,
        'parental_edu':parental_edu,'internet':internet,'health':health,
        'absences':absences,'final_score':final_score,
        'grade':[score_to_grade(s) for s in final_score],
        'result':['PASS' if s>=40 else 'FAIL' for s in final_score]
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"  Generated {len(df)} records → {DATA_PATH}")
else:
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} records from {DATA_PATH}")

print(f"\n  Shape     : {df.shape}")
print(f"  Grade dist:\n{df['grade'].value_counts().to_string()}")
print(f"\n  Pass/Fail :\n{df['result'].value_counts().to_string()}")
print(f"\n  Feature summary:\n{df[FEATURES].describe().round(2).to_string()}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: EDA Plots
# ═══════════════════════════════════════════════════════════════════
sep("STEP 2: EDA PLOTS")

# Plot 1 — Score distribution + grade bar
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
grade_order  = [g for g in ['O','A+','A','B+','B','C','F'] if g in df['grade'].values]
grade_counts = df['grade'].value_counts().reindex(grade_order, fill_value=0)
GRADE_COLOR  = {'O':'#1abc9c','A+':'#27ae60','A':'#2980b9','B+':'#8e44ad','B':'#f39c12','C':'#e67e22','F':'#e74c3c'}
colors_g = [GRADE_COLOR[g] for g in grade_order]

axes[0].bar(grade_counts.index, grade_counts.values, color=colors_g, edgecolor='white', linewidth=1.5)
axes[0].set_title('Grade Distribution', fontweight='bold')
axes[0].set_xlabel('Grade'); axes[0].set_ylabel('Count')
for i,(idx,val) in enumerate(grade_counts.items()):
    axes[0].text(i, val+1, str(val), ha='center', fontweight='bold')

axes[1].hist(df['final_score'], bins=25, color='#5c67f2', edgecolor='white', alpha=0.85)
axes[1].axvline(df['final_score'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df['final_score'].mean():.1f}")
axes[1].axvline(40, color='orange', linestyle=':', linewidth=2, label='Pass Line (40)')
axes[1].set_title('Final Score Distribution', fontweight='bold')
axes[1].legend()

corr_vals = df[FEATURES].corrwith(df['final_score']).sort_values(ascending=True)
colors_c = ['#27ae60' if v > 0 else '#e74c3c' for v in corr_vals]
axes[2].barh(corr_vals.index, corr_vals.values, color=colors_c, edgecolor='white', height=0.6)
axes[2].axvline(0, color='black', linewidth=0.8)
axes[2].set_title('Correlation with Final Score', fontweight='bold')
axes[2].set_xlabel('Pearson r')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_eda_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/01_eda_overview.png")

# Plot 2 — Correlation heatmap
corr = df[FEATURES + ['final_score']].corr()
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/02_correlation_heatmap.png")

# Plot 3 — Feature distributions
num_feats = ['attendance','prev_marks','study_hours','assignments','internals','absences']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
palette = ['#5c67f2','#27ae60','#e67e22','#8e44ad','#2980b9','#e74c3c']
for ax, feat, color in zip(axes.flat, num_feats, palette):
    ax.hist(df[feat], bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(df[feat].mean(), color='black', linestyle='--', linewidth=1.8,
               label=f"μ={df[feat].mean():.1f}")
    ax.set_title(feat.replace('_',' ').title(), fontweight='bold')
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/03_feature_distributions.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Preprocessing
# ═══════════════════════════════════════════════════════════════════
sep("STEP 3: PREPROCESSING")

X       = df[FEATURES]
y_score = df['final_score']
y_pass  = (df['final_score'] >= 40).astype(int)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

Xtr, Xte, ytr_s, yte_s = train_test_split(X_scaled, y_score, test_size=TEST_SIZE, random_state=RANDOM_STATE)
_,   _,   ytr_p, yte_p = train_test_split(X_scaled, y_pass,  test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"  Train samples : {len(Xtr)}")
print(f"  Test  samples : {len(Xte)}")
print(f"  Features      : {len(FEATURES)}")
print(f"  PASS in test  : {yte_p.sum()} / {len(yte_p)}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Train 6 Models
# ═══════════════════════════════════════════════════════════════════
sep("STEP 4: MODEL TRAINING")

results_reg, results_clf = {}, {}

# ── Regression Models ─────────────────────────────────────────────
print("\n  [ Regression Models ]")

lin_reg = LinearRegression()
lin_reg.fit(Xtr, ytr_s)
p_lr = np.clip(lin_reg.predict(Xte), 0, 100)
rmse_lr = np.sqrt(mean_squared_error(yte_s, p_lr))
r2_lr   = r2_score(yte_s, p_lr)
results_reg['Linear Regression']  = {'RMSE':round(rmse_lr,4), 'R2':round(r2_lr,4)}
print(f"  Linear Regression   | RMSE={rmse_lr:.4f} | R2={r2_lr:.4f}")

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10,
                                min_samples_split=5, min_samples_leaf=2,
                                random_state=RANDOM_STATE, n_jobs=-1)
rf_reg.fit(Xtr, ytr_s)
p_rfr = np.clip(rf_reg.predict(Xte), 0, 100)
rmse_rfr = np.sqrt(mean_squared_error(yte_s, p_rfr))
r2_rfr   = r2_score(yte_s, p_rfr)
results_reg['Random Forest Reg']  = {'RMSE':round(rmse_rfr,4), 'R2':round(r2_rfr,4)}
print(f"  Random Forest Reg   | RMSE={rmse_rfr:.4f} | R2={r2_rfr:.4f}")

gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=4, random_state=RANDOM_STATE)
gb_reg.fit(Xtr, ytr_s)
p_gbr = np.clip(gb_reg.predict(Xte), 0, 100)
rmse_gbr = np.sqrt(mean_squared_error(yte_s, p_gbr))
r2_gbr   = r2_score(yte_s, p_gbr)
results_reg['Gradient Boosting']  = {'RMSE':round(rmse_gbr,4), 'R2':round(r2_gbr,4)}
print(f"  Gradient Boosting   | RMSE={rmse_gbr:.4f} | R2={r2_gbr:.4f}")

# ── Classification Models ─────────────────────────────────────────
print("\n  [ Classification Models ]")

log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(Xtr, ytr_p)
p_log = log_reg.predict(Xte)
acc_log = accuracy_score(yte_p, p_log)
results_clf['Logistic Regression'] = {'Accuracy': round(acc_log*100, 2)}
print(f"  Logistic Regression | Accuracy={acc_log*100:.2f}%")

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)
rf_clf.fit(Xtr, ytr_p)
p_rfc = rf_clf.predict(Xte)
acc_rfc = accuracy_score(yte_p, p_rfc)
results_clf['Random Forest Clf']   = {'Accuracy': round(acc_rfc*100, 2)}
print(f"  Random Forest Clf   | Accuracy={acc_rfc*100:.2f}%")

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale',
                 probability=True, random_state=RANDOM_STATE)
svm_model.fit(Xtr, ytr_p)
p_svm = svm_model.predict(Xte)
acc_svm = accuracy_score(yte_p, p_svm)
results_clf['SVM (RBF)']           = {'Accuracy': round(acc_svm*100, 2)}
print(f"  SVM (RBF Kernel)    | Accuracy={acc_svm*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Evaluation Plots
# ═══════════════════════════════════════════════════════════════════
sep("STEP 5: EVALUATION PLOTS")

# Plot 4 — Model comparison charts
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_m = ['#5c67f2','#27ae60','#e67e22']

reg_names = list(results_reg.keys())
rmse_vals = [results_reg[m]['RMSE'] for m in reg_names]
r2_vals   = [results_reg[m]['R2']   for m in reg_names]

bars0 = axes[0].bar(reg_names, rmse_vals, color=colors_m, edgecolor='white', width=0.5, linewidth=1.5)
axes[0].set_title('RMSE (Lower = Better)', fontweight='bold')
axes[0].set_ylabel('RMSE')
for bar,val in zip(bars0, rmse_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f'{val:.3f}', ha='center', fontweight='bold')

bars1 = axes[1].bar(reg_names, r2_vals, color=colors_m, edgecolor='white', width=0.5, linewidth=1.5)
axes[1].set_title('R\u00b2 Score (Higher = Better)', fontweight='bold')
axes[1].set_ylabel('R\u00b2'); axes[1].set_ylim(0, 1)
for bar,val in zip(bars1, r2_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{val:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_regression_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/04_regression_comparison.png")

# Plot 5 — Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, pred, label, cmap in zip(axes,
                                  [p_log, p_rfc, p_svm],
                                  list(results_clf.keys()),
                                  ['Blues','Greens','Oranges']):
    cm = confusion_matrix(yte_p, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['FAIL','PASS'])
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f'{label}\nAcc: {accuracy_score(yte_p,pred)*100:.2f}%', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/05_confusion_matrices.png")

# Plot 6 — Feature importance
fi = pd.Series(rf_reg.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi)))
bars = axes[0].barh(fi.index, fi.values, color=colors_fi, edgecolor='white', height=0.6)
for bar, val in zip(bars, fi.values):
    axes[0].text(val+0.002, bar.get_y()+bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
axes[0].set_title('Feature Importance (RF Regressor)', fontweight='bold')
axes[0].set_xlabel('Importance Score')
axes[1].pie(fi.values[::-1], labels=fi.index[::-1], autopct='%1.1f%%', startangle=140,
             colors=plt.cm.Set3(np.linspace(0,1,len(fi))),
             wedgeprops={'edgecolor':'white','linewidth':1.5})
axes[1].set_title('Importance Share', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/06_feature_importance.png")

# Plot 7 — Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, pred, label, color in zip(axes,
                                   [p_lr, p_rfr, p_gbr],
                                   ['Linear Regression','Random Forest','Gradient Boosting'],
                                   ['#5c67f2','#27ae60','#e67e22']):
    ax.scatter(yte_s, pred, alpha=0.5, color=color, edgecolors='none', s=25)
    ax.plot([0,100],[0,100], 'r--', linewidth=2, label='Perfect')
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.set_xlabel('Actual Score'); ax.set_ylabel('Predicted Score')
    ax.set_title(f'{label}\nR\u00b2={r2_score(yte_s,pred):.4f}', fontweight='bold')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/07_actual_vs_predicted.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Cross-Validation
# ═══════════════════════════════════════════════════════════════════
sep("STEP 6: 5-FOLD CROSS VALIDATION")

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X_arr = np.array(X_scaled)

cv_results = {}
for name, model, y_target, scoring in [
    ('Linear Regression',   LinearRegression(), y_score, 'r2'),
    ('RF Regressor',        RandomForestRegressor(n_estimators=50,random_state=RANDOM_STATE), y_score, 'r2'),
    ('Gradient Boosting',   GradientBoostingRegressor(n_estimators=50,random_state=RANDOM_STATE), y_score, 'r2'),
    ('Logistic Regression', LogisticRegression(max_iter=500,random_state=RANDOM_STATE), y_pass, 'accuracy'),
    ('RF Classifier',       RandomForestClassifier(n_estimators=50,random_state=RANDOM_STATE), y_pass, 'accuracy'),
    ('SVM (RBF)',           SVC(kernel='rbf',random_state=RANDOM_STATE), y_pass, 'accuracy'),
]:
    scores = cross_val_score(model, X_arr, y_target, cv=kf, scoring=scoring)
    cv_results[name] = {'Mean': round(scores.mean(),4), 'Std': round(scores.std(),4)}
    print(f"  {name:<25} {scores.mean():.4f} +/- {scores.std():.4f}  ({scoring})")

# ═══════════════════════════════════════════════════════════════════
# STEP 7: Save Models
# ═══════════════════════════════════════════════════════════════════
sep("STEP 7: SAVING MODELS")

save_pkl(scaler,    'scaler.pkl')
save_pkl(lin_reg,   'lin_reg.pkl')
save_pkl(rf_reg,    'rf_reg.pkl')
save_pkl(gb_reg,    'gb_reg.pkl')
save_pkl(log_reg,   'log_reg.pkl')
save_pkl(rf_clf,    'rf_clf.pkl')
save_pkl(svm_model, 'svm.pkl')
save_pkl(fi,        'feature_importance.pkl')
save_pkl(confusion_matrix(yte_p, p_rfc), 'confusion_matrix.pkl')
save_pkl(classification_report(yte_p, p_rfc,
         target_names=['FAIL','PASS'], output_dict=True), 'clf_report.pkl')

# ═══════════════════════════════════════════════════════════════════
# STEP 8: Predict Function
# ═══════════════════════════════════════════════════════════════════
sep("STEP 8: PREDICTION FUNCTION")

def predict_student(attendance, prev_marks, study_hours, assignments,
                    internals, extracurricular=0, parental_edu=1,
                    internet=1, health=3, absences=4):
    """
    Predict student academic performance using ensemble of trained models.

    Parameters (all numeric):
        attendance      : 0-100  (attendance percentage)
        prev_marks      : 0-100  (previous semester marks %)
        study_hours     : 0-12   (daily study hours)
        assignments     : 0-100  (assignment completion %)
        internals       : 0-100  (internal exam marks %)
        extracurricular : 0 or 1 (0=No, 1=Yes)
        parental_edu    : 0,1,2  (0=None, 1=Graduate, 2=Postgrad)
        internet        : 0 or 1 (0=No, 1=Yes)
        health          : 1-5    (1=Poor to 5=Excellent)
        absences        : 0-30   (absent days)

    Returns:
        dict with predicted_score, grade, status, risk_level, pass_probability
    """
    input_data = pd.DataFrame([[attendance, prev_marks, study_hours, assignments,
                                 internals, extracurricular, parental_edu,
                                 internet, health, absences]], columns=FEATURES)
    scaled = scaler.transform(input_data)

    s_lr  = float(np.clip(lin_reg.predict(scaled)[0], 0, 100))
    s_rf  = float(np.clip(rf_reg.predict(scaled)[0],  0, 100))
    s_gb  = float(np.clip(gb_reg.predict(scaled)[0],  0, 100))
    avg   = round((s_lr + s_rf + s_gb) / 3, 1)

    prob_pass = float(rf_clf.predict_proba(scaled)[0][1]) * 100
    grade     = score_to_grade(avg)
    status    = 'PASS' if avg >= 40 else 'FAIL'
    risk, icon = get_risk(avg)

    return {
        'predicted_score':  avg,
        'score_lr':         round(s_lr, 1),
        'score_rf':         round(s_rf, 1),
        'score_gb':         round(s_gb, 1),
        'grade':            grade,
        'status':           status,
        'risk_level':       risk,
        'risk_icon':        icon,
        'pass_probability': round(prob_pass, 1)
    }

# ── Test Predictions ─────────────────────────────────────────────────────────
test_cases = [
    ("Arjun (Topper)",    90, 85, 7,   90, 82, 1, 2, 1, 4, 2),
    ("Priya (Average)",   74, 62, 3.5, 68, 58, 0, 1, 1, 3, 5),
    ("Rohit (At Risk)",   52, 38, 1.5, 42, 35, 0, 0, 0, 2, 12),
]

print(f"\n  {'STUDENT':<22} {'SCORE':>6} {'GRADE':>6} {'STATUS':>6} {'RISK':>8} {'PASS%':>7}")
print("  " + "-"*58)
for name, att, pm, sh, asgn, intl, ec, pe, inet, hlth, abs_d in test_cases:
    r = predict_student(att, pm, sh, asgn, intl, ec, pe, inet, hlth, abs_d)
    print(f"  {name:<22} {r['predicted_score']:>6.1f} {r['grade']:>6} "
          f"{r['status']:>6} {r['risk_icon']+r['risk_level']:>9} {r['pass_probability']:>6.1f}%")

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
sep("FINAL SUMMARY")

print("\n  REGRESSION MODELS:")
print(f"  {'Model':<22} {'RMSE':>8} {'R2':>8}")
print("  " + "-"*40)
for m, v in results_reg.items():
    print(f"  {m:<22} {v['RMSE']:>8.4f} {v['R2']:>8.4f}")

print("\n  CLASSIFICATION MODELS:")
print(f"  {'Model':<25} {'Accuracy':>10}")
print("  " + "-"*37)
for m, v in results_clf.items():
    print(f"  {m:<25} {v['Accuracy']:>9.2f}%")

print(f"\n  Models saved in  : ./{MODEL_DIR}/")
print(f"  Plots saved in   : ./{PLOT_DIR}/")
print(f"\n  FILES GENERATED:")
for f in sorted(os.listdir(MODEL_DIR)): print(f"    models/{f}")
for f in sorted(os.listdir(PLOT_DIR)):  print(f"    plots/{f}")
print("\n" + "="*60)
print("  TRAINING COMPLETE!")
print("="*60)
