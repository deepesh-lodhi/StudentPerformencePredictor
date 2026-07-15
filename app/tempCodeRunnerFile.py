def load_models():
    base=r'C:\Users\HP\OneDrive\Desktop\SSP_Project\StudentPerformancePredictor\spp\models' + '\\'
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

