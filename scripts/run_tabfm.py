import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src and tabfm to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tabfm")))

from src.preprocessing.Imputers.Imputer import Imputation
from src.utils.Writers.Writer import Writer

def evaluate(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        # Fallback if probability fails
        y_prob = y_pred

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y, y_prob)
    except Exception:
        pass
    
    cm = confusion_matrix(y, y_pred).ravel()
    if len(cm) == 4:
        tn, fp, fn, tp = cm
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    cm_values = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    
    return {
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_prob,
        "confusion_matrix": cm_values,
        "images": {} 
    }

def main():
    try:
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        model = tabfm_v1_0_0.load()
        clf = TabFMClassifier(model=model)
        has_tabfm = True
    except ImportError as e:
        print(f"TabFM import failed. Make sure you install its dependencies: {e}")
        print("To install: pip install ./tabfm[pytorch] safetensors huggingface_hub")
        has_tabfm = False

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "mimic_iii_processed.csv")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Assuming 'mortality_flag' is the target based on typical MIMIC datasets
    target_column = "mortality_flag"
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found.")
        return

    # Split into train, dev, test
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
    
    # Recreate dfs for imputation
    df_train = X_train.copy()
    df_train[target_column] = y_train
    
    df_dev = X_dev.copy()
    df_dev[target_column] = y_dev
    
    df_test = X_test.copy()
    df_test[target_column] = y_test
    
    print("Imputing missing values using IterativeImputer...")
    imputer = Imputation(method="iterative", random_state=42)
    imputer.fit(df_train)
    
    df_train_imp = imputer.transform(df_train)
    df_dev_imp = imputer.transform(df_dev)
    df_test_imp = imputer.transform(df_test)
    
    X_train_imp = df_train_imp.drop(columns=[target_column])
    y_train_imp = df_train_imp[target_column].values
    
    X_dev_imp = df_dev_imp.drop(columns=[target_column])
    y_dev_imp = df_dev_imp[target_column].values
    
    X_test_imp = df_test_imp.drop(columns=[target_column])
    y_test_imp = df_test_imp[target_column].values
    
    if not has_tabfm:
        print("Skipping TabFM modeling phase because it is not installed.")
        return
        
    print("Fitting TabFMClassifier...")
    clf.fit(X_train_imp, y_train_imp)
    
    print("Evaluating with TabFMClassifier...")
    validation_results = evaluate(clf, X_dev_imp, y_dev_imp)
    test_results = evaluate(clf, X_test_imp, y_test_imp)
    
    results = {
        "model": clf,
        "validation": validation_results,
        "test": test_results
    }
    
    print("Writing results using HTMLWriter...")
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    writer = Writer(output_dir=results_dir)
    out_path = writer.write_to_html(results, "TabFM_mimic_iii")
    print(f"Done! Results written to: {out_path}")

if __name__ == "__main__":
    main()
