from sklearn.experimental import enable_iterative_imputer


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



df = pd.read_csv('./data/mimic_iv_merged_data_cleaned.csv')

# remove some columns
columns_to_drop = [
    'temperature_avg',
    'map_avg',
    'dbil_avg',
    'bmi_avg',
    'fibrinogen_avg',
    'height_avg',
    'race',
    'insurance'
]

df = df.drop(columns = columns_to_drop)


# point, do label-encoding and stuff before other things. also 
# make sure to apply the same transformations to all datasets

# also put checks in all classes
# also do it step by step. Very important....


le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])


# X = df.drop('mortality_flag', axis=1)
# y = df['mortality_flag']

X = df.copy()

# do train val test split

X_train, X_test = train_test_split(X, test_size=0.15, random_state=42)

X_train, X_val = train_test_split(X_train, test_size=0.17, random_state=42)



df_iterative = Imputation.iterative(X_train, None)

# df_rf = Imputation.random_forest(X_train, None)


df_iterative = Smoother.winsorization(df_iterative)
# df_rf = Smoother.winsorization(df_rf)





df_iterative = Generator.SMOTE(df_iterative, 'mortality_flag')
# df_rf = Generator.SMOTE(df_rf, 'mortality_flag')





X_train_iterative = df_iterative.drop('mortality_flag', axis=1)
y_train_iterative = df_iterative['mortality_flag']

X_val_iterative = X_val.drop('mortality_flag', axis = 1)
y_val_iterative = X_val['mortality_flag']

X_test_iterative = X_test.drop('mortality_flag', axis = 1)
y_test_iterative = X_test['mortality_flag']

models = Models()

results_iterative = models.xgboost(
    X_train_iterative,
    y_train_iterative,
    X_val_iterative,
    y_val_iterative,
    X_test_iterative,
    y_test_iterative
)


writer = Writer()

print(writer.write_to_md(results_iterative, 'xgboost_iterative_results'))