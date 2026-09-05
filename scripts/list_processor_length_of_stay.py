from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split

import os
from src.preprocessing.Imputers import Imputation
from src.preprocessing.Smoother import Smoother
from src.modeling.Models.Models_regression import ModelsRegression
from src.utils.Writers import Writer
from src.preprocessing.Outlier import Outlier



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


##############################################

# The data we need to concern ourselves with

csv1_path = './data/mimic_iv_processed_length_of_stay.csv'
csv2_path = './data/mimic_iii_processed_length_of_stay.csv'

# Target column for this regression task
TARGET_COLUMN = "length_of_stay_hours"

# Smooth out extra outliers
Smoothers_list = ["winsorization"]

# Impute missing values
Imputers_list = ["iterative", "knn"]

# At this point, we do the train / val / test split

# NOTE: Generators (SMOTE, CTGAN, etc.) are designed for *classification*
#       (class imbalance). They are NOT applicable to a continuous regression
#       target such as length_of_stay_hours, so no Generators_list is defined.

# AI models to train and test on the data (regression variants)
Models_list = [
    "Catboost",
    "XGBoost",
    "randomforest",
    "decisiontree",
    "linear_regression",
    "lightgbm",
]

# Write methods
Writers_list = ["write_to_md", "write_to_html", "write_to_csv"]

#################################################




###### STEP 1: Import data ##############

df = pd.read_csv(csv1_path)
df_validation = pd.read_csv(csv2_path)

#########################################




######STEP 2: Applying Outlier removal ##########################

df_outlier_removed = {}

columns_outlier = [
    "age",
    "rbc",
    "wbc",
    "hgb",
    "plt",
    "rdw",
    "hct",
    "aptt",
    "pt",
    "inr",
    "bicarbonate",
    "base_excess",
    "anion_gap",
    "chloride",
    "calcium",
    "sodium",
    "potassium",
    "glucose",
    "creatinine",
    "bun",
    "tbil",
    "albumin",
    "alt",
    "ast",
    "alp",
    "fibrinogen",
    "dbil",
    "temperature",
    "map",
    "sbp",
    "dbp",
    "heart_rate",
    "respiratory_rate",
    "o2",
    "weight",
    "height",
    "bmi",
    # Also winsorise the target itself so extreme stays don't dominate
    TARGET_COLUMN,
]

# Check if Outliers_list is defined and not empty, otherwise default to raw data
if 'Outliers_list' in globals() and Outliers_list:
    for outlier in Outliers_list:
        df_outlier_removed[outlier] = getattr(Outlier, outlier)(df, columns_outlier)
else:
    print("[OUTLIER] No outlier removal method specified. Using raw data.")
    df_outlier_removed = {"raw": df}

#########################################




########STEP 3: Do train/val/test split BEFORE smoothing/imputation #########################
# CRITICAL: Split BEFORE any imputation to prevent data leakage.

df_train_split = {}
df_val_split = {}
df_test_split = {}

for outlier_method, df_out_rem in df_outlier_removed.items():
    print(f"[SPLIT] Starting split for: {outlier_method}")
    print(f"[SPLIT] Total rows: {len(df_out_rem)}, Unique patients: {len(df_out_rem['subject_id'].unique())}")

    # Get unique patients
    patient_ids = df_out_rem["subject_id"].unique()

    # First split: 70% train, 30% temporary
    train_ids, temp_ids = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=42
    )

    # Second split: 15% validation, 15% test (split the 30% equally)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42
    )

    # Create the actual DataFrames based on patient IDs
    train_df = df_out_rem[df_out_rem["subject_id"].isin(train_ids)].copy()
    val_df = df_out_rem[df_out_rem["subject_id"].isin(val_ids)].copy()
    test_df = df_out_rem[df_out_rem["subject_id"].isin(test_ids)].copy()

    print(f"[SPLIT] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    df_train_split[outlier_method] = train_df
    df_val_split[outlier_method] = val_df
    df_test_split[outlier_method] = test_df

#########################################




########STEP 4: Smoothening & Imputation (fit on train, apply to val/test) #################################

df_smoothed = {}

for outlier_method in df_train_split.keys():
    print(f"[PROCESS] Starting processing for outlier method: {outlier_method}")

    # Get train, val, test for this outlier method
    df_train = df_train_split[outlier_method].copy()
    df_val = df_val_split[outlier_method].copy()
    df_test = df_test_split[outlier_method].copy()

    # Save subject_id and target column
    train_subject_ids = df_train["subject_id"].copy()
    train_target_values = df_train[TARGET_COLUMN].copy()
    
    val_subject_ids = df_val["subject_id"].copy()
    val_target_values = df_val[TARGET_COLUMN].copy()
    
    test_subject_ids = df_test["subject_id"].copy()
    test_target_values = df_test[TARGET_COLUMN].copy()

    # Remove subject_id and target before processing
    df_train_for_processing = df_train.drop(columns=["subject_id", TARGET_COLUMN])
    df_val_for_processing = df_val.drop(columns=["subject_id", TARGET_COLUMN])
    df_test_for_processing = df_test.drop(columns=["subject_id", TARGET_COLUMN])

    for smoother in Smoothers_list:
        print(f"[SMOOTH] Running smoother: {smoother}")
        
        # Apply smoother to each split
        df_train_smoothed = getattr(Smoother, smoother)(df=df_train_for_processing)
        df_val_smoothed = getattr(Smoother, smoother)(df=df_val_for_processing)
        df_test_smoothed = getattr(Smoother, smoother)(df=df_test_for_processing)

        for imputer in Imputers_list:
            print(f"[IMPUTE] Running imputation: {imputer} on smoothed data")
            
            # Create imputer instance and fit ONLY on train data
            imputer_instance = Imputation(imputer)
            imputer_instance.fit(df_train_smoothed)

            # Transform all three sets using the same fitted imputer
            df_train_result = imputer_instance.transform(df_train_smoothed)
            df_val_result = imputer_instance.transform(df_val_smoothed)
            df_test_result = imputer_instance.transform(df_test_smoothed)

            # Add subject_id and target back
            df_train_result.insert(0, "subject_id", train_subject_ids.values)
            df_train_result.insert(3, TARGET_COLUMN, train_target_values.values)
            
            df_val_result.insert(0, "subject_id", val_subject_ids.values)
            df_val_result.insert(3, TARGET_COLUMN, val_target_values.values)
            
            df_test_result.insert(0, "subject_id", test_subject_ids.values)
            df_test_result.insert(3, TARGET_COLUMN, test_target_values.values)

            # Store generated dataframes
            method_key = f"{outlier_method}_{smoother}_{imputer}"
            df_smoothed[method_key] = (df_train_result, df_val_result, df_test_result)
            print(f"[PROCESS] Completed {method_key}")

#########################################




########STEP 5: Extract train/val/test from processed data #################################

df_train = {}
df_val = {}
df_test = {}

for method, (train_df, val_df, test_df) in df_smoothed.items():
    print(f"[EXTRACT] Extracting splits for: {method}")
    print(f"[EXTRACT] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Store them
    df_train[method] = train_df
    df_val[method]   = val_df
    df_test[method]  = test_df

#########################################




# ##### Step 6: Train & evaluate regression models ############################################

results_dir = "../results_length_of_stay"
os.makedirs(results_dir, exist_ok=True)

# Create Writer instance
writer = Writer(output_dir=results_dir)

for method, train_df in df_train.items():
    print(f"[MODEL] Training model for: {method}")
    val_df  = df_val[method]
    test_df = df_test[method]

    # Feature columns: everything except subject_id and the target
    feature_columns = [
        col for col in train_df.columns
        if col not in ["subject_id", TARGET_COLUMN]
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_val   = val_df[feature_columns]
    y_val   = val_df[TARGET_COLUMN]
    X_test  = test_df[feature_columns]
    y_test  = test_df[TARGET_COLUMN]

    # Prepare external dataset (df_validation)
    X_external = df_validation[feature_columns]
    y_external = df_validation[TARGET_COLUMN]

    print(f"[MODEL] X_train shape: {X_train.shape}, y_train mean: {y_train.mean():.2f}")

    # Train and evaluate each regression model
    for model_name in Models_list:
        full_name = f"{method}_{model_name}"
        print(f"[MODEL] Training {model_name} for {method}")

        # Create ModelsRegression instance
        models_instance = ModelsRegression(results_dir=results_dir)

        # Dispatch to the correct regression method
        if model_name.lower() == "catboost":
            results = models_instance.catboost(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                X_external, y_external
            )
        elif model_name.lower() == "xgboost":
            results = models_instance.xgboost(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                X_external, y_external
            )
        elif model_name.lower() in ["randomforest", "random_forest"]:
            results = models_instance.random_forest(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                X_external, y_external
            )
        elif model_name.lower() in ["decisiontree", "decision_tree"]:
            results = models_instance.decision_tree(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                X_external, y_external
            )
        elif model_name.lower() in ["linear_regression", "linearregression"]:
            results = models_instance.linear_regression(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                X_external, y_external
            )
        elif model_name.lower() == "lightgbm":
            results = models_instance.lightgbm(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                X_external, y_external
            )
        else:
            print(f"[MODEL] Unknown model '{model_name}', skipping.")
            continue

        print(f"[MODEL] Completed {model_name} for {method}")

        # Write results
        for writer_method in Writers_list:
            writer_func = getattr(writer, writer_method)
            writer_func(results, full_name)

# ######################################################################
