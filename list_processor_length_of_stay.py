from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split

import os
from Imputers import Imputation
from Smoother import Smoother
from Models.Models_regression import ModelsRegression
from Writers import Writer
from Outlier import Outlier



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




########STEP 3: Smoothening & Imputation (Smoothening -> Imputation) #################################

df_smoothed = {}

for outlier_method, df_out_rem in df_outlier_removed.items():
    print(f"[PROCESS] Starting processing for outlier method: {outlier_method}")

    # Save subject_id and target column
    subject_ids = df_out_rem["subject_id"].copy()
    target_values = df_out_rem[TARGET_COLUMN].copy()

    # Remove subject_id and target before processing
    df_for_processing = df_out_rem.drop(
        columns=["subject_id", TARGET_COLUMN]
    )

    for smoother in Smoothers_list:
        print(f"[SMOOTH] Running smoother: {smoother}")
        # Run smoother on the feature-only DataFrame
        df_smoothed_features = getattr(Smoother, smoother)(df=df_for_processing)

        for imputer in Imputers_list:
            print(f"[IMPUTE] Running imputation: {imputer} on smoothed data")
            # Run imputation (target_column hint is used internally by some
            # imputers to avoid leaking the target; pass it so the API is
            # consistent even though target has been removed from df)
            df_result = getattr(Imputation, imputer)(
                df=df_smoothed_features,
                target_column=TARGET_COLUMN
            )

            # Add subject_id and target back
            df_result.insert(0, "subject_id", subject_ids.values)
            df_result.insert(3, TARGET_COLUMN, target_values.values)

            # Store generated dataframe in df_smoothed
            method_key = f"{outlier_method}_{smoother}_{imputer}"
            df_smoothed[method_key] = df_result
            print(f"[PROCESS] Completed {method_key}, rows: {len(df_result)}")


#########################################




########STEP 4: Train / Val / Test split #################################

df_train = {}
df_val = {}
df_test = {}

for method, df in df_smoothed.items():
    print(f"[SPLIT] Starting split for: {method}")
    print(f"[SPLIT] Total rows: {len(df)}, Unique patients: {len(df['subject_id'].unique())}")

    # Get unique patients
    patient_ids = df["subject_id"].unique()

    # First split: 70 % train, 30 % temporary
    train_ids, temp_ids = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=42
    )

    # Second split: 15 % validation, 15 % test
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42
    )

    # Create the actual DataFrames
    train_df = df[df["subject_id"].isin(train_ids)].copy()
    val_df   = df[df["subject_id"].isin(val_ids)].copy()
    test_df  = df[df["subject_id"].isin(test_ids)].copy()

    print(f"[SPLIT] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Store them
    df_train[method] = train_df
    df_val[method]   = val_df
    df_test[method]  = test_df

#########################################




# ##### Step 5: Train & evaluate regression models ############################################

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
