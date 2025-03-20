import pandas as pd
from typing import Tuple

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets from given paths
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        
    Returns:
        Tuple containing train and test DataFrames
    """
    try:
        loan_train = pd.read_csv(train_path)
        loan_test = pd.read_csv(test_path)
        return loan_train, loan_test
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def combine_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine train and test datasets into single DataFrame
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Combined DataFrame
    """
    return pd.concat([train_df, test_df], ignore_index=True) 