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

# Leave this empty to use raw data. Add methods and matching flags to enable
# outlier processing, e.g. ["iqr", "modified_z_score"].
Outliers_list = []
Outliers_run = []

# Smooth out extra outlier
Smoothers_list = ["winsorization"]
Smoothers_run = [True]

# Impute missing values
Imputers_list = ["iterative", "knn"]
Imputers_run = [True, True]

# At this point, we do the train val test split

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


validate_run_flags(Imputers_list, Imputers_run, "Imputers")
validate_run_flags(Smoothers_list, Smoothers_run, "Smoothers")
validate_run_flags(Generators_list, Generators_run, "Generators")
validate_run_flags(Outliers_list, Outliers_run, "Outliers")






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

# Check if Outliers_list is defined and not empty, otherwise default to using raw data
if Outliers_list:
    for outlier, should_run in zip(Outliers_list, Outliers_run):
        if not should_run:
            continue
        df_outlier_removed[outlier] = getattr(Outlier, outlier)(df, columns_outlier)
else:
    print("[OUTLIER] No outlier removal method specified. Using raw data.")
    df_outlier_removed = {"raw": df}

#########################################








########STEP 3: Smoothening & Imputation (Smoothening -> Imputation) #################################

df_smoothed = {}

for outlier_method, df_out_rem in df_outlier_removed.items():
    print(f"[PROCESS] Starting processing for outlier method: {outlier_method}")

    # Save subject_id and mortality_flag
    subject_ids = df_out_rem["subject_id"].copy()
    mortality = df_out_rem["mortality_flag"].copy()

    # Remove subject_id and mortality_flag before processing
    df_for_processing = df_out_rem.drop(
        columns=["subject_id", "mortality_flag"]
    )

    for smoother, smoother_run in zip(Smoothers_list, Smoothers_run):
        if not smoother_run:
            continue
        print(f"[SMOOTH] Running smoother: {smoother}")
        # Run smoother on the feature-only DataFrame
        df_smoothed_features = getattr(Smoother, smoother)(df=df_for_processing)

        for imputer, imputer_run in zip(Imputers_list, Imputers_run):
            if not imputer_run:
                continue
            print(f"[IMPUTE] Running imputation: {imputer} on smoothed data")
            # Run imputation
            df_result = getattr(Imputation, imputer)(
                df=df_smoothed_features,
                target_column="mortality_flag"
            )

            # Add subject_id and mortality_flag back
            df_result.insert(0, "subject_id", subject_ids)
            df_result.insert(3, "mortality_flag", mortality)

            # Store generated dataframe in df_smoothed
            method_key = f"{outlier_method}_{smoother}_{imputer}"
            df_smoothed[method_key] = df_result
            print(f"[PROCESS] Completed {method_key}, rows: {len(df_result)}")


#########################################




########STEP 5: do a train/val/test split #################################

df_train = {}
df_val = {}
df_test = {}
dataset_saver = DatasetSaver(
    output_dir=os.path.join("saved_datasets", ACTIVE_FEATURE_PROJECT)
)

for method, df in df_smoothed.items():
    print(f"[SPLIT] Starting split for: {method}")
    print(f"[SPLIT] Total rows: {len(df)}, Unique patients: {len(df['subject_id'].unique())}")

    # Get unique patients
    patient_ids = df["subject_id"].unique()

    # First split: 70% train, 30% temporary
    train_ids, temp_ids = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=42
    )

    # Second split: 15% validation, 15% test
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42
    )

    # Create the actual DataFrames
    train_df = df[df["subject_id"].isin(train_ids)].copy()
    val_df = df[df["subject_id"].isin(val_ids)].copy()
    test_df = df[df["subject_id"].isin(test_ids)].copy()

    print(f"[SPLIT] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Store them
    df_train[method] = train_df
    df_val[method] = val_df
    df_test[method] = test_df
    dataset_saver.save(method, train_df, test_df, val_df)

# A false preprocessing flag means use the already-saved complete triplet for
# that method instead of rerunning its preprocessing steps.
outlier_cache_methods = Outliers_list or ["raw"]
outlier_cache_flags = Outliers_run or [True]
for outlier_method, outlier_run in zip(outlier_cache_methods, outlier_cache_flags):
    for smoother, smoother_run in zip(Smoothers_list, Smoothers_run):
        for imputer, imputer_run in zip(Imputers_list, Imputers_run):
            method = f"{outlier_method}_{smoother}_{imputer}"
            if outlier_run and smoother_run and imputer_run:
                continue
            cached = dataset_saver.load(method)
            df_train[method] = cached["train"]
            df_test[method] = cached["test"]
            df_val[method] = cached["val"]

#########################################




# ##### Step 6: Apply generators on Train #############################

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




# ##### Step 7: Apply the AI model ############################################################

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

        print(f"[MODEL] Completed {model_name} for {method}")

        # Call writer methods
        for writer_method in Writers_list:
            writer_func = getattr(writer, writer_method)
            writer_func(results, full_name)

# ######################################################################

