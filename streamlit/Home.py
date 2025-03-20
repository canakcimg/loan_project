import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from src
from src.data_import import load_data

# Set page config
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Loan Prediction Application")
st.markdown("""
This app provides tools to analyze loan data and predict loan approval status.
            
### Features:
- **Loan Dataset**: View and explore the loan dataset
- **EDA & Visualization**: Explore data analysis and visualizations
            
Use the sidebar to navigate between different pages.
""")



# Show basic statistics
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(parent_dir, "data/train.csv")
    test_path = os.path.join(parent_dir, "data/test.csv")
    train_df, test_df = load_data(train_path, test_path)
    
    st.subheader("Quick Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Training data: {train_df.shape[0]} records with {train_df.shape[1]} features")
    with col2:
        st.info(f"Testing data: {test_df.shape[0]} records with {test_df.shape[1]} features")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Loan Prediction System | Created with Streamlit") 