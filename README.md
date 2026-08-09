# Research Speedrun: Clinical Prediction Pipeline

A robust, modular machine learning pipeline designed to process clinical data (based on MIMIC-III and MIMIC-IV datasets) to predict clinical outcomes such as mortality.

---

## 📂 Project Structure

```bash
Research Speedrun/
├── data/                       # Dataset directory
│   ├── mimic_iii_processed.csv # Validation/external dataset
│   └── mimic_iv_processed.csv  # Primary training/development dataset
├── EDA/                        # Exploratory Data Analysis
│   └── CSV_file_maker.ipynb    # Data exploration & initial preparation
├── Outlier/                    # Outlier handling module
│   ├── Outlier.py              # Outlier detection methods (IQR, Modified Z-score)
│   └── __init__.py
├── Imputers/                   # Missing data imputation module
│   ├── Imputer.py              # Imputation wrappers (Iterative, etc.)
│   └── __init__.py
├── Smoother/                   # Feature smoothing module
│   ├── Smoother.py             # Winsorization smoothing to clip remaining outliers
│   └── __init__.py
├── Generators/                 # Resampling & data balancing module
│   ├── Generator.py            # Oversampling (SMOTE) and Downsampling methods
│   └── __init__.py
├── Models/                     # ML Model implementations
│   ├── Models.py               # CatBoost and XGBoost training and evaluation
│   └── __init__.py
├── Writers/                    # Output formatting and export module
│   ├── Writer.py               # Exports results to Markdown, HTML, and CSV
│   └── __init__.py
├── list_processor.py           # Main orchestration script
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation (this file)
```

---

## ⚙️ ML Pipeline Flow

The execution in `list_processor.py` follows a sequential multi-stage process:

```mermaid
graph TD
    A[Import MIMIC-IV & MIMIC-III Data] --> B[Outlier Removal (IQR / Modified Z-Score)]
    B --> C[Imputation (Iterative Imputer)]
    C --> D[Smoothening (Winsorization)]
    D --> E[Train / Val / Test Split]
    E --> F[Class Balancing (SMOTE / Downsampler)]
    F --> G[Model Training & Evaluation (CatBoost / XGBoost)]
    G --> H[Export Results (MD, HTML, CSV)]
```

---

## 🚀 How to Run

1. **Activate the Virtual Environment**:
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the Pipeline**:
   ```bash
   python list_processor.py
   ```

---

## 🖥️ GPU Support (Automatic Detection)

To utilize GPU resources if available, the classifiers support native hardware acceleration.

### 1. CatBoost GPU configuration:
Add `task_type="GPU"` to the instantiation.
```python
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    task_type="GPU",
    verbose=False
)
```

### 2. XGBoost GPU configuration:
Add `device="cuda"` (XGBoost 2.0+) or `tree_method="hist", device="cuda"` to the instantiation.
```python
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    device="cuda"
)
```

### Automatic Implementation Example in `Models/Models.py`
```python
from catboost.utils import get_gpu_device_count

# Check if GPU is present
has_gpu = False
try:
    has_gpu = get_gpu_device_count() > 0
except Exception:
    pass

# For CatBoost
params = {"iterations": 500, "learning_rate": 0.05}
if has_gpu:
    params["task_type"] = "GPU"

# For XGBoost
xgb_params = {"n_estimators": 500, "learning_rate": 0.05}
if has_gpu:
    xgb_params["device"] = "cuda"
```

---

## 🔍 Hyperparameter Tuning: Grid-Search CV

To perform Grid Search Cross-Validation (Grid-Search CV) for the models, wrap them using Scikit-Learn's `GridSearchCV`.

### Implementation Example

Here is how you can implement `GridSearchCV` in the pipeline:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 1. Define parameter grid for tuning
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5]
}

# 2. Instantiate base estimator
base_model = CatBoostClassifier(
    iterations=500, 
    verbose=False,
    task_type="GPU" if has_gpu else "CPU"
)

# 3. Setup GridSearchCV
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,  # Use all CPU cores for parallel tuning
    verbose=1
)

# 4. Fit on training data
grid_search.fit(X_train, y_train)

# 5. Extract the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
```

---

## 📊 Dataset Columns Reference

The processed dataset contains the following variables:

| # | Column Name | Non-Null Count | Dtype | Description / Category |
|---|-------------|----------------|-------|------------------------|
| 0 | sex | 666 | str | Demographics |
| 1 | age | 666 | int64 | Demographics |
| 2 | race | 666 | str | Demographics |
| 3 | insurance | 666 | str | Demographics |
| 4 | myocardial_infarction | 666 | int64 | Comorbidities |
| 5 | congestive_heart_failure | 666 | int64 | Comorbidities |
| 6 | peripheral_vascular_disease | 666 | int64 | Comorbidities |
| 7 | cerebrovascular_disease | 666 | int64 | Comorbidities |
| 8 | dementia | 666 | int64 | Comorbidities |
| 9 | chronic_pulmonary_disease | 666 | int64 | Comorbidities |
| 10 | rheumatic_disease | 666 | int64 | Comorbidities |
| 11 | peptic_ulcer_disease | 666 | int64 | Comorbidities |
| 12 | diabetes | 666 | int64 | Comorbidities |
| 13 | paraplegia | 666 | int64 | Comorbidities |
| 14 | renal_disease | 666 | int64 | Comorbidities |
| 15 | malignant_cancer | 666 | int64 | Comorbidities |
| 16 | severe_liver_disease | 666 | int64 | Comorbidities |
| 17 | aids | 666 | int64 | Comorbidities |
| 18 | rbc_avg | 484 | float64 | Lab Values |
| 19 | wbc_avg | 630 | float64 | Lab Values |
| 20 | hgb_avg | 645 | float64 | Lab Values |
| 21 | plt_avg | 645 | float64 | Lab Values |
| 22 | rdw_avg | 642 | float64 | Lab Values |
| 23 | hct_avg | 649 | float64 | Lab Values |
| 24 | aptt_avg | 593 | float64 | Lab Values |
| 25 | pt_avg | 612 | float64 | Lab Values |
| 26 | inr_avg | 611 | float64 | Lab Values |
| 27 | bicarbonate_avg | 648 | float64 | Lab Values |
| 28 | lactate_avg | 114 | float64 | Lab Values |
| 29 | base_excess_avg | 302 | float64 | Lab Values |
| 30 | anion_gap_avg | 651 | float64 | Lab Values |
| 31 | chloride_avg | 626 | float64 | Lab Values |
| 32 | calcium_avg | 582 | float64 | Lab Values |
| 33 | sodium_avg | 523 | float64 | Lab Values |
| 34 | potassium_avg | 570 | float64 | Lab Values |
| 35 | glucose_avg | 652 | float64 | Lab Values |
| 36 | creatinine_avg | 449 | float64 | Lab Values |
| 37 | bun_avg | 649 | float64 | Lab Values |
| 38 | tbil_avg | 575 | float64 | Lab Values |
| 39 | albumin_avg | 450 | float64 | Lab Values |
| 40 | alt_avg | 562 | float64 | Lab Values |
| 41 | ast_avg | 566 | float64 | Lab Values |
| 42 | alp_avg | 562 | float64 | Lab Values |
| 43 | fibrinogen_avg | 160 | float64 | Lab Values |
| 44 | dbil_avg | 112 | float64 | Lab Values |
| 45 | temperature_avg | 0 | float64 | Vitals |
| 46 | map_avg | 293 | float64 | Vitals |
| 47 | sbp_avg | 455 | float64 | Vitals |
| 48 | dbp_avg | 456 | float64 | Vitals |
| 49 | heart_rate_avg | 452 | float64 | Vitals |
| 50 | respiratory_rate_avg | 456 | float64 | Vitals |
| 51 | o2_avg | 457 | float64 | Vitals |
| 52 | weight_avg | 380 | float64 | Physical metrics |
| 53 | height_avg | 58 | float64 | Physical metrics |
| 54 | bmi_avg | 47 | float64 | Physical metrics |
| 55 | mortality_flag | 666 | int64 | **Target Variable** |