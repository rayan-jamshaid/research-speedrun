Run the list_processor2.py here.

This one is somewhat like this

##############################################

# The data we need to concern ourselves with

csv1_path = './data/mimic_iv_processed.csv'
csv2_path = './data/mimic_iii_processed.csv'


# Smooth out extra outlier
Smoothers_list = ["winsorization"]

# Impute missing values
Imputers_list = ["iterative", "knn"]

# At this point, we do the train val test split

# Generator to generate synthetic data, remove class imbalance
Generators_list = ["SMOTE", "downsampler"]

# AI models to train and test on the data
Models_list = ["Catboost", "XGBoost"]

# Write methods
Writers_list = ["write_to_md", "write_to_html", "write_to_csv"]

#################################################



List Processor 1 also works as intended.


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


