from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split

import os
from Imputers import Imputation
from Smoother import Smoother
from Generators import Generator
from Models import Models
from Writers import Writer
from Outlier import Outlier



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


##############################################

# The data we need to concern ourselves with

csv1_path = './data/mimic_iv_processed.csv'
csv2_path = './data/mimic_iii_processed.csv'

# remove the outliers
Outliers_list = ["iqr", "modified_z_score"]

# Impute missing values
Imputers_list = ["iterative", "knn"]

# Smooth out extra outlier
Smoothers_list = ["winsorization"]

# At this point, we do the train val test split

# Generator to generate synthetic data, remove class imbalance
Generators_list = ["SMOTE", "downsampler", "CTGAN", "CopulaGAN", "TVAE"]

# AI models to train and test on the data
Models_list = ["Catboost", "XGBoost", "randomforest", "decisiontree", "logistic_regression", "lightgbm"]

# Write methods
# Writers_list = ["write_to_csv"]

Writers_list = ["write_to_md", "write_to_html", "write_to_csv"]

#################################################






###### STEP 1: Import data##############


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
    "bmi"
]

for outlier in Outliers_list:
    df_outlier_removed[outlier] = getattr(Outlier, outlier)(df, columns_outlier)

#########################################








########STEP 3: Impute#################################

df_imputed = {}

for outlier_method, df_out_rem in df_outlier_removed.items():
    print(f"[IMPUTE] Starting imputation for outlier method: {outlier_method}")

    # Save subject_id and mortality_flag
    subject_ids = df_out_rem["subject_id"].copy()
    mortality = df_out_rem["mortality_flag"].copy()

    # Remove subject_id and mortality_flag before imputation
    df_for_imputation = df_out_rem.drop(
        columns=["subject_id", "mortality_flag"]
    )

    for imputer in Imputers_list:
        print(f"[IMPUTE] Running imputation: {imputer}")

        # Run imputation
        df_result = getattr(Imputation, imputer)(
            df=df_for_imputation,
            target_column="mortality_flag"
        )

        # Add subject_id and mortality_flag back
        df_result.insert(0, "subject_id", subject_ids)
        df_result.insert(3, "mortality_flag", mortality)

        # Store generated dataframe
        df_imputed[f"{outlier_method}_{imputer}"] = df_result
        print(f"[IMPUTE] Completed imputation: {imputer}, rows: {len(df_result)}")


#########################################





########STEP 4: Smoothening################################

df_smoothed = {}

for imputer_method, df_imp in df_imputed.items():
    print(f"[SMOOTH] Starting smoothing for: {imputer_method}")
    for smoother in Smoothers_list:
        print(f"[SMOOTH] Running smoother: {smoother}")
        df_smoothed[f"{imputer_method}_{smoother}"] = getattr(Smoother, smoother)(df = df_imp)
        print(f"[SMOOTH] Completed smoother: {smoother}, rows: {len(df_smoothed[f'{imputer_method}_{smoother}'])}")


#########################################




########STEP 5: do a train/val/test split #################################

df_train = {}
df_val = {}
df_test = {}

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

#########################################




# ##### Step 6: Apply generators on Train #############################

df_trained_generated = {}

for name, df_tr in df_train.items():
    print(f"[GENERATOR] Starting generation for: {name}, rows: {len(df_tr)}")

    for generator in Generators_list:
        print(f"[GENERATOR] Running generator: {generator}")
        df_trained_generated[
            f"{name}_{generator}"
        ] = getattr(Generator, generator)(
            df_tr,
            "mortality_flag"
        )
        print(f"[GENERATOR] Completed generator: {generator}, rows: {len(df_trained_generated[f'{name}_{generator}'])}")

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

        print(f"[MODEL] Completed {model_name} for {method}")

        # Call writer methods
        for writer_method in Writers_list:
            writer_func = getattr(writer, writer_method)
            writer_func(results, full_name)

# For datasets with generators: use df_trained_generated
# Generators are applied on df_train, so we need to split again for val/test
for method, df in df_trained_generated.items():
    print(f"[MODEL] Training model for generated: {method}, rows: {len(df)}")
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

