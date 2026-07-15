# 🎓 Student Performance Predictor
## Minor Project II — CSE (Data Science) | RGPV Bhopal | 2024-25

---

## 📁 Project Structure

```
StudentPerformancePredictor/
│
├── model.py                          ← Complete ML training script (RUN THIS FIRST)
├── app/app.py                        ← Streamlit web application
├── Student_Performance_Predictor.ipynb ← Jupyter Notebook (full EDA + training)
│
├── data/
│   └── student_data.csv             ← Dataset (auto-generated if missing)
│
├── models/                          ← Saved .pkl model files (auto-created)
│   ├── scaler.pkl
│   ├── lin_reg.pkl
│   ├── rf_reg.pkl
│   ├── gb_reg.pkl
│   ├── log_reg.pkl
│   ├── rf_clf.pkl
│   ├── svm.pkl
│   └── ...
│
├── plots/                           ← EDA & evaluation plots (auto-created)
│   ├── 01_eda_overview.png
│   ├── 02_correlation_heatmap.png
│   └── ...
│
└── requirements.txt                 ← Python dependencies
```

---

## ⚙️ Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train ML models
```bash
python model.py
```
This will:
- Generate `student_data.csv` (if not present)
- Train 6 ML models
- Save all `.pkl` files to `models/`
- Save 7 evaluation plots to `plots/`
- Print full training summary

### Step 3 — Run Streamlit App
```bash
cd app
streamlit run app.py
```
Open: http://localhost:8501

### Step 4 — Open Jupyter Notebook (optional)
```bash
pip install notebook
jupyter notebook Student_Performance_Predictor.ipynb
```

---

## 🤖 ML Models Trained

| # | Model | Task | Best Metric |
|---|-------|------|-------------|
| 1 | Linear Regression | Score Prediction | R² ≈ 0.76 |
| 2 | Random Forest Regressor | Score Prediction | RMSE ≈ 7.8 |
| 3 | Gradient Boosting Regressor | Score Prediction | RMSE ≈ 7.5 |
| 4 | Logistic Regression | Pass/Fail | Acc ≈ 93.75% |
| 5 | Random Forest Classifier | Pass/Fail | Acc ≈ 93.75% |
| 6 | SVM (RBF Kernel) | Pass/Fail | Acc ≈ 93.75% |

**Ensemble Score** = Average of Linear Reg + RF Reg + Gradient Boosting

---

## 📊 Features Used

| Feature | Description | Type |
|---------|-------------|------|
| attendance | Attendance percentage | Numeric |
| prev_marks | Previous semester marks % | Numeric |
| study_hours | Daily study hours | Numeric |
| assignments | Assignment completion % | Numeric |
| internals | Internal exam marks % | Numeric |
| extracurricular | Activity participation | Binary (0/1) |
| parental_edu | Parental education level | Ordinal (0/1/2) |
| internet | Internet access | Binary (0/1) |
| health | Health status | Ordinal (1-5) |
| absences | Number of absent days | Numeric |

---

## 🏆 Grade Scale (RGPV)

| Grade | Marks Range |
|-------|------------|
| O | ≥ 90% |
| A+ | 80-89% |
| A | 70-79% |
| B+ | 60-69% |
| B | 50-59% |
| C | 40-49% |
| F | < 40% (FAIL) |

---

## 👨‍💻 Team

- **NAME-1** — Enrollment No. 1
- **NAME-2** — Enrollment No. 2
- **NAME-3** — Enrollment No. 3

**Guide:** [Guide Name] | **Co-Guide:** [Co-Guide Name]

**School of Information Technology**
Rajiv Gandhi Proudyogiki Vishwavidyalaya, Bhopal — 2024-25
