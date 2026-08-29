# Feature slicer

`FeatureSlicer` runs the complete pipeline on a named set of input features.
It always retains `subject_id` and `mortality_flag`, so patient-level splitting
and model labels continue to work.

## Configure an experiment

At the top of either processor, configure a project and select it:

```python
FEATURE_PROJECTS = {
    "all_features": None,
    "test_project1": ["age", "sex", "rbc", "wbc", "hgb", "plt",
                      "creatinine", "bun", "heart_rate", "respiratory_rate"],
    "test_project2": ["albumin", "alt", "ast", "alp", "fibrinogen",
                      "dbil", "temperature", "map", "weight", "height"],
}
ACTIVE_FEATURE_PROJECT = "test_project1"
```

Replace the example columns with the feature sets you want to compare, such
as your 10 most-important or 10 least-important features. Every selected
feature must exist in **both** input CSV files.

## Run and reuse data

The processor slices both source and external-validation CSVs before every
downstream stage. Snapshots are kept separate by feature project:

```text
saved_datasets/test_project1/iqr_iterative_winsorization_CTGAN/
  iqr_iterative_winsorization_CTGAN_train.csv
  iqr_iterative_winsorization_CTGAN_test.csv
  iqr_iterative_winsorization_CTGAN_val.csv
```

Set a corresponding `*_run` flag to `False` to reuse its snapshot.
