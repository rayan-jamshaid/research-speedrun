import pandas as pd
from typing import List

class FeatureSlicer:
    """
    A class to slice a DataFrame and return only the specified features.
    """
    
    def __init__(self, features: List[str] = None):
        """
        Initialize the slicer.
        
        Args:
            features (List[str], optional): Default list of features to slice. 
                                            Can be overridden when calling the slice method.
        """
        self.features = features

    def slice(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """
        Returns a new DataFrame containing only the specified features.
        
        Args:
            df (pd.DataFrame): The input pandas DataFrame.
            features (List[str], optional): List of columns to keep. If not provided, 
                                            it uses the features initialized with the class.
                                            
        Returns:
            pd.DataFrame: A DataFrame with only the selected features.
        """
        features_to_slice = features if features is not None else self.features
        
        if features_to_slice is None:
            raise ValueError("No features provided to slice. Please provide a list of features.")
            
        # Optional: verify all features exist in the DataFrame to provide a clearer error message
        missing_features = [f for f in features_to_slice if f not in df.columns]
        if missing_features:
            raise KeyError(f"The following features were not found in the DataFrame: {missing_features}")
            
        return df[features_to_slice]
