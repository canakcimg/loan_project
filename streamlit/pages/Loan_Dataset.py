import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now import from src
from src.data_import import load_data, combine_data
from src.data_analysis import check_df, grab_col_names, missing_values_table

# Page config
st.set_page_config(
    page_title="Loan Dataset",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Loan Dataset Explorer")
st.markdown("View and explore the loan datasets used for prediction.")

# Load data
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(parent_dir, "data/train.csv")
    test_path = os.path.join(parent_dir, "data/test.csv")
    train_df, test_df = load_data(train_path, test_path)
    combined_df = combine_data(train_df, test_df)
    
    # Sidebar for options
    st.sidebar.header("Dataset Options")
    dataset_option = st.sidebar.radio(
        "Select Dataset to View",
        ["Training Data", "Testing Data", "Combined Data"]
    )
    
    # Display based on selection
    if dataset_option == "Training Data":
        st.subheader("Training Dataset")
        if st.checkbox("Show Training Data Shape"):
            st.write(f"Shape: {train_df.shape[0]} rows and {train_df.shape[1]} columns")
        
        if st.checkbox("Show Training Data Head"):
            num_rows = st.slider("Number of rows to view", 5, 50, 10)
            st.dataframe(train_df.head(num_rows))
            
        if st.checkbox("Show Summary Statistics"):
            st.write(train_df.describe())
            
        if st.checkbox("Show Column Information"):
            st.write(train_df.info())
            
    elif dataset_option == "Testing Data":
        st.subheader("Testing Dataset")
        if st.checkbox("Show Testing Data Shape"):
            st.write(f"Shape: {test_df.shape[0]} rows and {test_df.shape[1]} columns")
        
        if st.checkbox("Show Testing Data Head"):
            num_rows = st.slider("Number of rows to view", 5, 50, 10)
            st.dataframe(test_df.head(num_rows))
            
        if st.checkbox("Show Summary Statistics"):
            st.write(test_df.describe())
            
    else:  # Combined Data
        st.subheader("Combined Dataset")
        if st.checkbox("Show Combined Data Shape"):
            st.write(f"Shape: {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")
        
        if st.checkbox("Show Combined Data Head"):
            num_rows = st.slider("Number of rows to view", 5, 50, 10)
            st.dataframe(combined_df.head(num_rows))
        
        if st.checkbox("Show Summary Statistics"):
            st.write(combined_df.describe())
            
        if st.checkbox("Show Missing Values"):
            missing_df = missing_values_table(combined_df)
            st.dataframe(missing_df)
            
    # Feature selection
    st.sidebar.header("Feature Analysis")
    if st.sidebar.checkbox("Analyze Selected Feature"):
        # Get column names
        cat_cols, num_cols, cat_but_car = grab_col_names(combined_df)
        
        # Select feature
        all_cols = combined_df.columns.tolist()
        selected_feature = st.sidebar.selectbox("Select Feature", all_cols)
        
        st.subheader(f"Analysis of {selected_feature}")
        
        # Display feature statistics
        feature_data = combined_df[selected_feature]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Feature Statistics:")
            st.write(f"Count: {feature_data.count()}")
            st.write(f"Unique Values: {feature_data.nunique()}")
            st.write(f"Missing Values: {feature_data.isna().sum()}")
        
        with col2:
            if selected_feature in num_cols:
                st.write("Numeric Feature Details:")
                st.write(f"Mean: {feature_data.mean()}")
                st.write(f"Median: {feature_data.median()}")
                st.write(f"Std Dev: {feature_data.std()}")
                st.write(f"Min: {feature_data.min()}")
                st.write(f"Max: {feature_data.max()}")
            else:
                st.write("Categorical Feature Details:")
                st.write(feature_data.value_counts())
        
        # Visualization for the feature
        st.subheader(f"Visualization for {selected_feature}")
        
        if selected_feature in num_cols:
            # For numeric features
            hist_values = np.histogram(
                feature_data.dropna(), bins=30, range=(feature_data.min(), feature_data.max())
            )[0]
            st.bar_chart(hist_values)
        else:
            # For categorical features
            st.bar_chart(feature_data.value_counts())

except Exception as e:
    st.error(f"Error loading or processing data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Loan Prediction System | Dataset Explorer") 