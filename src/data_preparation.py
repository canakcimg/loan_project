import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List

def handle_outliers(dataframe: pd.DataFrame, 
                   col_name: str, 
                   q1: float = 0.05, 
                   q3: float = 0.95) -> Tuple[float, float]:
    """
    Calculate outlier thresholds and handle outliers
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe: pd.DataFrame, variable: str) -> None:
    """
    Replace outliers with threshold values
    """
    low_limit, up_limit = handle_outliers(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = dataframe[variable].dtype.type(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = dataframe[variable].dtype.type(up_limit)

def one_hot_encoder(dataframe: pd.DataFrame, 
                   categorical_cols: List[str], 
                   drop_first: bool = False) -> pd.DataFrame:
    """
    Perform one-hot encoding on categorical variables
    
    Args:
        dataframe: Input DataFrame
        categorical_cols: List of categorical column names
        drop_first: Whether to drop first dummy variable
        
    Returns:
        DataFrame with one-hot encoded features
    """
    # Remove target variable from categorical columns if present
    categorical_cols = [col for col in categorical_cols if col != 'loan_status']
    
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

def prepare_data(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Prepare data for modeling by handling outliers and scaling
    """
    # Handle outliers
    for col in num_cols:
        replace_with_thresholds(df, col)
    
    # Scale numerical features
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df 