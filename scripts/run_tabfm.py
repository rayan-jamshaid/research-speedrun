import os
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

def run_dataset(data_path, dataset_name, device):
    """Run the complete TabFM workflow for one processed MIMIC dataset."""
    try:
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        model = tabfm_v1_0_0.load(device=device)
        clf = TabFMClassifier(model=model)
    except ImportError as e:
        print(f"TabFM import failed. Make sure its dependencies are installed: {e}")
        print("To install: pip install ./tabfm[pytorch] safetensors huggingface_hub")
        return

    print(f"\n[TABFM] Processing {dataset_name}: {data_path}")
    df = pd.read_csv(data_path)
    target_column = "mortality_flag"
    if target_column not in df.columns:
        print(f"[SKIP] Target column '{target_column}' not found in {data_path}.")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
    )

    print(f"[SPLIT] {dataset_name}: train={len(X_train)}, dev={len(X_dev)}, test={len(X_test)}")
    print(f"[IMPUTE] {dataset_name}: IterativeImputer")
    imputer = Imputation(method="iterative", random_state=42)
    # Fit on training features only: target labels must never influence
    # imputation of validation or test features.
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_dev_imp = imputer.transform(X_dev)
    X_test_imp = imputer.transform(X_test)
    y_train_imp = y_train.values
    y_dev_imp = y_dev.values
    y_test_imp = y_test.values

    print(f"[TABFM] Fitting {dataset_name}...")
    clf.fit(X_train_imp, y_train_imp)
    results = {
        "model": clf,
        "validation": evaluate(clf, X_dev_imp, y_dev_imp),
        "test": evaluate(clf, X_test_imp, y_test_imp),
    }
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    out_path = Writer(output_dir=results_dir).write_to_html(results, f"TabFM_{dataset_name}")
    print(f"[DONE] {dataset_name}: results written to {out_path}")


def main():
    # Detect GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[DEVICE] GPU detected: {torch.cuda.get_device_name(0)} — using CUDA.")
        else:
            device = "cpu"
            print("[DEVICE] No GPU detected — using CPU.")
    except ImportError:
        device = "cpu"
        print("[DEVICE] PyTorch not available for device detection — defaulting to CPU.")

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    for dataset_name in ("mimic_iii", "mimic_iv"):
        data_path = os.path.join(data_dir, f"{dataset_name}_processed.csv")
        if not os.path.exists(data_path):
            print(f"[SKIP] Missing dataset: {data_path}")
            continue
        run_dataset(data_path, dataset_name, device)

if __name__ == "__main__":
    main()
