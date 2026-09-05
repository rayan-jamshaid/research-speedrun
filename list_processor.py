from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split

import os
from Imputers import Imputation
from Smoother import Smoother
from Generators import Generator
from Models import Models
from Writers import Writer
from Outlier import Outlier
from DatasetSaver import DatasetSaver
from features_slicer import FeatureSlicer



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


##############################################

# The data we need to concern ourselves with

csv1_path = './data/mimic_iv_processed.csv'
csv2_path = './data/mimic_iii_processed.csv'

# Add named experiments here. None keeps every feature from the raw CSV.
FEATURE_PROJECTS = {
    "all_features": None,
    "test_project1": ["age", "sex", "rbc", "wbc", "hgb", "plt", "creatinine", "bun", "heart_rate", "respiratory_rate"],
    "test_project2": ["albumin", "alt", "ast", "alp", "fibrinogen", "dbil", "temperature", "map", "weight", "height"],
}
ACTIVE_FEATURE_PROJECT = "all_features"

# remove the outliers
Outliers_list = ["iqr", "modified_z_score"]
Outliers_run = [True, True]

# Impute missing values
Imputers_list = ["iterative", "knn"]
Imputers_run = [True, True]

# Smooth out extra outlier
Smoothers_list = ["winsorization"]
Smoothers_run = [True]

# Generator to generate synthetic data, remove class imbalance
Generators_list = ["SMOTE", "downsampler", "CTGAN", "CopulaGAN", "TVAE"]
Generators_run = [True, True, True, True, True]

# AI models to train and test on the data
Models_list = ["Catboost", "XGBoost", "randomforest", "decisiontree", "logistic_regression", "lightgbm"]

# Write methods
# Writers_list = ["write_to_csv"]

Writers_list = ["write_to_md", "write_to_html", "write_to_csv"]

#################################################

def validate_run_flags(methods, run_flags, label):
    if len(methods) != len(run_flags):
        raise ValueError(f"{label}_run must have one boolean for every {label} method.")


validate_run_flags(Outliers_list, Outliers_run, "Outliers")
validate_run_flags(Imputers_list, Imputers_run, "Imputers")
validate_run_flags(Smoothers_list, Smoothers_run, "Smoothers")
validate_run_flags(Generators_list, Generators_run, "Generators")




###### STEP 1: Import data##############

df = pd.read_csv(csv1_path)
df_validation = pd.read_csv(csv2_path)

feature_slicer = FeatureSlicer(FEATURE_PROJECTS)
df, df_validation = feature_slicer.slice_pair(
    df, df_validation, ACTIVE_FEATURE_PROJECT
)

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
    "bmi"
]
columns_outlier = [column for column in columns_outlier if column in df.columns]

for outlier, should_run in zip(Outliers_list, Outliers_run):
    if not should_run:
        continue
    df_outlier_removed[outlier] = getattr(Outlier, outlier)(df, columns_outlier)

#########################################




########STEP 3: Do train/val/test split BEFORE imputation #########################
# CRITICAL: Split BEFORE imputation to prevent data leakage.
# Imputer will ONLY be fit on training data, then applied to val/test.

df_train_by_outlier = {}
df_val_by_outlier = {}
df_test_by_outlier = {}

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

    df_train_by_outlier[outlier_method] = train_df
    df_val_by_outlier[outlier_method] = val_df
    df_test_by_outlier[outlier_method] = test_df

#########################################




########STEP 4: Impute (fit on train, apply to val/test) ##############################

df_train_imputed = {}
df_val_imputed = {}
df_test_imputed = {}

for outlier_method in df_train_by_outlier.keys():
    print(f"[IMPUTE] Starting imputation for outlier method: {outlier_method}")
    
    # Get train, val, test for this outlier method
    df_train = df_train_by_outlier[outlier_method].copy()
    df_val = df_val_by_outlier[outlier_method].copy()
    df_test = df_test_by_outlier[outlier_method].copy()

    # Save subject_id and mortality_flag
    train_subject_ids = df_train["subject_id"].copy()
    train_mortality = df_train["mortality_flag"].copy()
    
    val_subject_ids = df_val["subject_id"].copy()
    val_mortality = df_val["mortality_flag"].copy()
    
    test_subject_ids = df_test["subject_id"].copy()
    test_mortality = df_test["mortality_flag"].copy()

    # Remove subject_id and mortality_flag before imputation
    df_train_for_imputation = df_train.drop(columns=["subject_id", "mortality_flag"])
    df_val_for_imputation = df_val.drop(columns=["subject_id", "mortality_flag"])
    df_test_for_imputation = df_test.drop(columns=["subject_id", "mortality_flag"])

    for imputer_method, should_run in zip(Imputers_list, Imputers_run):
        if not should_run:
            continue
        print(f"[IMPUTE] Running imputation: {imputer_method}")

        # Create imputer instance and fit ONLY on train data
        imputer = Imputation(imputer_method)
        imputer.fit(df_train_for_imputation)

        # Transform all three sets using the same fitted imputer
        df_train_result = imputer.transform(df_train_for_imputation)
        df_val_result = imputer.transform(df_val_for_imputation)
        df_test_result = imputer.transform(df_test_for_imputation)

        # Add subject_id and mortality_flag back
        df_train_result.insert(0, "subject_id", train_subject_ids)
        df_train_result.insert(3, "mortality_flag", train_mortality)
        
        df_val_result.insert(0, "subject_id", val_subject_ids)
        df_val_result.insert(3, "mortality_flag", val_mortality)
        
        df_test_result.insert(0, "subject_id", test_subject_ids)
        df_test_result.insert(3, "mortality_flag", test_mortality)

        # Store generated dataframes
        method_key = f"{outlier_method}_{imputer_method}"
        df_train_imputed[method_key] = df_train_result
        df_val_imputed[method_key] = df_val_result
        df_test_imputed[method_key] = df_test_result
        
        print(f"[IMPUTE] Completed imputation: {imputer_method}")
        print(f"[IMPUTE]   Train rows: {len(df_train_result)}, Val rows: {len(df_val_result)}, Test rows: {len(df_test_result)}")

#########################################




########STEP 5: Smoothening ################################

df_train_smoothed = {}
df_val_smoothed = {}
df_test_smoothed = {}

for method_key in df_train_imputed.keys():
    print(f"[SMOOTH] Starting smoothing for: {method_key}")
    
    df_train = df_train_imputed[method_key]
    df_val = df_val_imputed[method_key]
    df_test = df_test_imputed[method_key]
    
    for smoother, should_run in zip(Smoothers_list, Smoothers_run):
        if not should_run:
            continue
        print(f"[SMOOTH] Running smoother: {smoother}")
        
        # Apply smoother to each split
        df_train_smoothed_result = getattr(Smoother, smoother)(df=df_train)
        df_val_smoothed_result = getattr(Smoother, smoother)(df=df_val)
        df_test_smoothed_result = getattr(Smoother, smoother)(df=df_test)
        
        method_full_key = f"{method_key}_{smoother}"
        df_train_smoothed[method_full_key] = df_train_smoothed_result
        df_val_smoothed[method_full_key] = df_val_smoothed_result
        df_test_smoothed[method_full_key] = df_test_smoothed_result
        
        print(f"[SMOOTH] Completed smoother: {smoother}")
        print(f"[SMOOTH]   Train rows: {len(df_train_smoothed_result)}, Val rows: {len(df_val_smoothed_result)}, Test rows: {len(df_test_smoothed_result)}")

#########################################




########STEP 6: Cache smoothed datasets and do train/val/test structure ########

df_train = {}
df_val = {}
df_test = {}
dataset_saver = DatasetSaver(
    output_dir=os.path.join("saved_datasets", ACTIVE_FEATURE_PROJECT)
)

for method in df_train_smoothed.keys():
    print(f"[CACHE] Saving smoothed dataset for: {method}")
    
    train_df = df_train_smoothed[method]
    val_df = df_val_smoothed[method]
    test_df = df_test_smoothed[method]
    
    df_train[method] = train_df
    df_val[method] = val_df
    df_test[method] = test_df
    
    dataset_saver.save(method, train_df, test_df, val_df)

# A false preprocessing flag means use the already-saved complete triplet for
# that method instead of rerunning its preprocessing steps.
for outlier, outlier_run in zip(Outliers_list, Outliers_run):
    for imputer, imputer_run in zip(Imputers_list, Imputers_run):
        for smoother, smoother_run in zip(Smoothers_list, Smoothers_run):
            method = f"{outlier}_{imputer}_{smoother}"
            if outlier_run and imputer_run and smoother_run:
                continue
            cached = dataset_saver.load(method)
            df_train[method] = cached["train"]
            df_test[method] = cached["test"]
            df_val[method] = cached["val"]

#########################################




# ##### Step 7: Apply generators on Train #############################

df_trained_generated = {}
generated_val = {}
generated_test = {}

for name, df_tr in df_train.items():
    print(f"[GENERATOR] Starting generation for: {name}, rows: {len(df_tr)}")

    for generator, should_run in zip(Generators_list, Generators_run):
        generated_name = f"{name}_{generator}"
        if not should_run:
            cached = dataset_saver.load(generated_name)
            df_trained_generated[generated_name] = cached["train"]
            generated_test[generated_name] = cached["test"]
            generated_val[generated_name] = cached["val"]
            print(f"[GENERATOR] Loaded saved data for: {generated_name}")
            continue
        print(f"[GENERATOR] Running generator: {generator}")
        df_trained_generated[
            generated_name
        ] = getattr(Generator, generator)(
            df_tr,
            "mortality_flag"
        )
        generated_train = df_trained_generated[generated_name]
        # Generators must only change training data; keep the held-out splits intact.
        generated_val[generated_name] = df_val[name]
        generated_test[generated_name] = df_test[name]
        dataset_saver.save(
            generated_name,
            generated_train,
            generated_test[generated_name],
            generated_val[generated_name],
        )
        print(f"[GENERATOR] Completed generator: {generator}, rows: {len(generated_train)}")

# ######################################################################




# ##### Step 8: Apply the AI model ############################################################

results_dir = "../results"
os.makedirs(results_dir, exist_ok=True)

# Create Writer instance
writer = Writer(output_dir=results_dir)

# For datasets without generators: use df_train, df_val, df_test
for method, train_df in df_train.items():
    print(f"[MODEL] Training model for: {method}")
    val_df = df_val[method]
    test_df = df_test[method]

    # Prepare external dataset (df_validation)
    feature_columns = [col for col in train_df.columns if col not in ["subject_id", "mortality_flag"]]
    X_train = train_df[feature_columns]
    y_train = train_df["mortality_flag"]
    X_val = val_df[feature_columns]
    y_val = val_df["mortality_flag"]
    X_test = test_df[feature_columns]
    y_test = test_df["mortality_flag"]

    # Prepare external data
    X_external = df_validation[feature_columns]
    y_external = df_validation["mortality_flag"]

    print(f"[MODEL] X_train shape: {X_train.shape}, y_train unique: {y_train.unique().tolist()}")

    # Train and evaluate each model
    for model_name in Models_list:
        # Build full method name: outlier_imputer_smoother_model
        full_name = f"{method}_{model_name}"
        print(f"[MODEL] Training {model_name} for {method}")

        # Create Models instance
        models_instance = Models(results_dir=results_dir)

        # Call method on instance with external data
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

        print(f"[MODEL] Completed {model_name} for {method}")

        # Call writer methods
        for writer_method in Writers_list:
            writer_func = getattr(writer, writer_method)
            writer_func(results, full_name)

# For generated datasets, use the generated training data with the original
# held-out validation and test sets. These are the exact triplets saved above.
for method, df in df_trained_generated.items():
    print(f"[MODEL] Training model for generated: {method}, rows: {len(df)}")
    train_df = df
    val_df = generated_val[method]
    test_df = generated_test[method]

    # Prepare data for training
    feature_columns = [col for col in df.columns if col not in ["subject_id", "mortality_flag"]]
    X_train = train_df[feature_columns]
    y_train = train_df["mortality_flag"]
    X_val = val_df[feature_columns]
    y_val = val_df["mortality_flag"]
    X_test = test_df[feature_columns]
    y_test = test_df["mortality_flag"]

    # Prepare external data
    X_external = df_validation[feature_columns]
    y_external = df_validation["mortality_flag"]

    print(f"[MODEL] X_train shape: {X_train.shape}, y_train unique: {y_train.unique().tolist()}")

    # Train and evaluate each model
    for model_name in Models_list:
        # Build full method name: outlier_imputer_smoother_generator_model
        full_name = f"{method}_{model_name}"
        print(f"[MODEL] Training {model_name} for {method}")

        # Create Models instance
        models_instance = Models(results_dir=results_dir)

        # Call method on instance with external data
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

        print(f"[MODEL] Completed {model_name} for {method}")

        # Call writer methods
        for writer_method in Writers_list:
            writer_func = getattr(writer, writer_method)
            writer_func(results, full_name)

# ######################################################################
