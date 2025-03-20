import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now import from src
from src.data_import import load_data, combine_data
from src.data_analysis import check_df, grab_col_names
from src.data_visualization import plot_numerical_dist, plot_correlation_matrix, plot_categorical_dist

# Configure page
st.set_page_config(
    page_title="EDA & Visualization",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("Exploratory Data Analysis & Visualization")
st.markdown("Explore the loan data through interactive visualizations.")

# Load data
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(parent_dir, "data/train.csv")
    test_path = os.path.join(parent_dir, "data/test.csv")
    train_df, test_df = load_data(train_path, test_path)
    df = combine_data(train_df, test_df)
    
    # Get column types
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    
    # Sidebar options
    st.sidebar.header("Visualization Options")
    viz_type = st.sidebar.radio(
        "Select Visualization Type",
        ["Numerical Distributions", "Categorical Distributions", "Correlation Analysis", "Custom Plots"]
    )
    
    # Display based on selection
    if viz_type == "Numerical Distributions":
        st.subheader("Numerical Feature Distributions")
        
        # Select numerical features to visualize
        selected_num_cols = st.multiselect(
            "Select Numerical Features",
            num_cols,
            default=num_cols[:min(3, len(num_cols))]
        )
        
        if selected_num_cols:
            # Set up figure
            fig, axes = plt.subplots(len(selected_num_cols), 2, figsize=(15, 5 * len(selected_num_cols)))
            
            # For a single feature, axes won't be 2D
            if len(selected_num_cols) == 1:
                axes = np.array([axes])
            
            for i, col in enumerate(selected_num_cols):
                # Histogram
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0])
                axes[i, 0].set_title(f'Distribution of {col}')
                
                # Box plot
                sns.boxplot(y=df[col].dropna(), ax=axes[i, 1])
                axes[i, 1].set_title(f'Boxplot of {col}')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Additional statistics
            st.subheader("Numerical Feature Statistics")
            st.dataframe(df[selected_num_cols].describe())
            
    elif viz_type == "Categorical Distributions":
        st.subheader("Categorical Feature Distributions")
        
        # Remove target variable from categorical features
        plot_cat_cols = [col for col in cat_cols if col != 'loan_status']
        
        # Select categorical features to visualize
        selected_cat_cols = st.multiselect(
            "Select Categorical Features",
            plot_cat_cols,
            default=plot_cat_cols[:min(3, len(plot_cat_cols))]
        )
        
        if selected_cat_cols:
            # For each selected categorical feature
            for col in selected_cat_cols:
                st.subheader(f"Distribution of {col}")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = df[col].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                plt.xticks(rotation=45)
                plt.title(f'Distribution of {col}')
                st.pyplot(fig)
                
                # Show value counts as a table
                st.write(f"Value counts for {col}:")
                st.write(pd.DataFrame({
                    'Values': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / value_counts.values.sum() * 100).round(2)
                }))
                
                st.markdown("---")
            
    elif viz_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        # Show correlation matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Select numerical features for correlation
        selected_corr_cols = st.multiselect(
            "Select Features for Correlation Analysis",
            num_cols,
            default=num_cols[:min(8, len(num_cols))]
        )
        
        if selected_corr_cols:
            # Calculate correlation matrix
            corr_matrix = df[selected_corr_cols].corr()
            
            # Plot heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                        cmap='coolwarm', ax=ax, linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Features')
            st.pyplot(fig)
            
            # Show correlations with target variable if available
            if 'loan_status' in df.columns and df['loan_status'].notna().any():
                st.subheader("Correlation with Target Variable (Loan Status)")
                
                # Create a DataFrame with target variable for correlation
                df_with_target = df[df['loan_status'].notna()]
                
                # Calculate correlations with target
                correlations = df_with_target[selected_corr_cols].corrwith(df_with_target['loan_status']).sort_values(ascending=False)
                
                # Plot correlations
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=correlations.index, y=correlations.values, ax=ax)
                plt.xticks(rotation=45)
                plt.title('Feature Correlation with Loan Status')
                st.pyplot(fig)
                
                # Show as table
                st.write("Correlation values with Loan Status:")
                st.dataframe(pd.DataFrame({
                    'Feature': correlations.index,
                    'Correlation': correlations.values.round(3)
                }).sort_values('Correlation', ascending=False))
            
    else:  # Custom Plots
        st.subheader("Custom Visualization")
        
        # Allow user to create custom plots
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter Plot", "Bar Chart", "Line Chart", "Pair Plot", "Box Plot"]
        )
        
        if chart_type == "Scatter Plot":
            # Select x and y variables
            x_var = st.selectbox("Select X-axis Variable", num_cols)
            y_var = st.selectbox("Select Y-axis Variable", [col for col in num_cols if col != x_var])
            
            # Optional color variable
            use_color = st.checkbox("Add Color Variable")
            color_var = None
            if use_color:
                color_var = st.selectbox("Select Color Variable", cat_cols)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            if use_color:
                scatter = sns.scatterplot(x=df[x_var], y=df[y_var], hue=df[color_var], ax=ax)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax)
            
            plt.title(f'Scatter Plot: {y_var} vs {x_var}')
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            st.pyplot(fig)
            
        elif chart_type == "Bar Chart":
            # Select categorical variable
            cat_var = st.selectbox("Select Categorical Variable", cat_cols)
            
            # Optional: Aggregate numerical variable
            agg_option = st.checkbox("Aggregate Numerical Variable")
            if agg_option:
                num_var = st.selectbox("Select Numerical Variable to Aggregate", num_cols)
                agg_func = st.selectbox("Select Aggregation Function", ["mean", "sum", "count", "min", "max"])
                
                # Create aggregated bar chart
                grouped_data = df.groupby(cat_var)[num_var].agg(agg_func).reset_index()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bar = sns.barplot(x=cat_var, y=num_var, data=grouped_data, ax=ax)
                plt.title(f'{agg_func.capitalize()} of {num_var} by {cat_var}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                # Simple value counts bar chart
                value_counts = df[cat_var].value_counts()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bar = sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                plt.title(f'Count of {cat_var}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
        elif chart_type == "Line Chart":
            # For line charts, we'd typically need time series data
            # Let's use this for numeric cumulative distributions instead
            num_var = st.selectbox("Select Numerical Variable", num_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            df[num_var].dropna().sort_values().reset_index(drop=True).plot(ax=ax)
            plt.title(f'Distribution of {num_var} (Sorted)')
            plt.xlabel('Index')
            plt.ylabel(num_var)
            st.pyplot(fig)
            
        elif chart_type == "Pair Plot":
            # Select a subset of numerical variables
            pair_vars = st.multiselect(
                "Select Variables for Pair Plot (3-5 recommended)",
                num_cols,
                default=num_cols[:min(3, len(num_cols))]
            )
            
            # Optional color variable
            use_color = st.checkbox("Add Color Variable to Pair Plot")
            color_var = None
            if use_color:
                color_var = st.selectbox("Select Color Variable for Pair Plot", cat_cols)
                pair_plot = sns.pairplot(df[pair_vars + [color_var]].dropna(), 
                                        hue=color_var, corner=True)
            else:
                pair_plot = sns.pairplot(df[pair_vars].dropna(), corner=True)
            
            plt.suptitle('Pair Plot of Selected Variables', y=1.02)
            st.pyplot(pair_plot)
            
        elif chart_type == "Box Plot":
            # Select numerical and categorical variables
            num_var = st.selectbox("Select Numerical Variable for Box Plot", num_cols)
            cat_var = st.selectbox("Select Categorical Variable for Grouping", cat_cols)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            box = sns.boxplot(x=cat_var, y=num_var, data=df, ax=ax)
            plt.title(f'Box Plot of {num_var} by {cat_var}')
            plt.xticks(rotation=45)
            st.pyplot(fig)

except Exception as e:
    st.error(f"Error processing data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Loan Prediction System | EDA & Visualization") 