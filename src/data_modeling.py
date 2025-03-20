from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
#from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

def split_data(X: pd.DataFrame, 
               y: pd.Series, 
               test_size: float = 0.20) -> Tuple:
    """
    Split data into training and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    classifiers = [
        ("CART", DecisionTreeClassifier(random_state=0)),
        ("RF", RandomForestClassifier(random_state=0, max_features='sqrt')),
        ('GBM', GradientBoostingClassifier(max_depth=4, random_state=0)),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(random_state=0, verbose=-1)),
        #('CatBoost', CatBoostClassifier(verbose=False))
    ]
    
    return X_train, X_test, y_train, y_test, classifiers

def evaluate_models(classifiers: List, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series) -> None:
    """
    Evaluate models using confusion matrix and classification report
    """
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        # Print classification report
        print(f'Classification Report for {name}:\n')
        print(classification_report(y_test, y_pred)) 