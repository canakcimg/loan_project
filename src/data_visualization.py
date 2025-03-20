import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import pandas as pd

def plot_numerical_dist(df: pd.DataFrame, numerical_cols: List[str]) -> None:
    """
    Plot distribution of numerical variables
    """
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        df[col].plot(kind='hist', bins=20, title=col)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame, numerical_cols: List[str]) -> None:
    """
    Plot correlation matrix for numerical variables
    """
    plt.figure(figsize=(20, 5))
    sns.heatmap(df[numerical_cols].corr(), annot=True)
    plt.show()

def plot_categorical_dist(df: pd.DataFrame, categorical_cols: List[str]) -> None:
    """
    Plot distribution of categorical variables
    """
    for col in categorical_cols:
        plt.figure(figsize=(8, 8))
        df.groupby(col).size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.title(f'Distribution of {col}')
        plt.show() 