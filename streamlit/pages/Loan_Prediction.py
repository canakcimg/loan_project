import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set page config
st.set_page_config(
    page_title="Loan Prediction",
    page_icon="🔮",
    layout="wide"
)

# Title
st.title("Loan Default Risk Prediction")
st.markdown("Enter loan applicant details to predict loan default risk.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(parent_dir, "models/LightGBM_bestmodel.pkl")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("Model loaded successfully!")
            return model
        else:
            st.warning(f"Model file not found at {model_path}. Will use a dummy model for demonstration.")
            # Create a dummy model for demonstration
            from lightgbm import LGBMClassifier
            dummy_model = LGBMClassifier(random_state=42)
            # Fit with some dummy data
            dummy_X = pd.DataFrame({
                'feature1': [0, 1], 
                'feature2': [1, 0]
            })
            dummy_y = pd.Series([0, 1])
            dummy_model.fit(dummy_X, dummy_y)
            return dummy_model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function for one-hot encoding (from notebook)
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Function to make predictions
def predict_loan_approval(features_df):
    model = load_model()
    if model is not None:
        # Ensure feature ordering matches model's expectations
        # For a real application, you would need to confirm the exact feature order
        
        try:
            prediction = model.predict(features_df)
            probability = model.predict_proba(features_df)
            return prediction[0], probability[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None
    return None, None

# Create the main layout with tabs
tab1, tab2 = st.tabs(["Input Data", "Results"])

with tab1:
    # Create two columns layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Applicant Information")
        
        # Personal Information (from notebook variables)
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        
        # Home ownership (categorical)
        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["MORTGAGE", "RENT", "OWN", "OTHER"]
        )
        
        # Default history (categorical)
        cb_person_default_on_file = st.selectbox(
            "Default on File",
            ["N", "Y"]
        )
        
        # Credit history length
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=6)

    with col2:
        st.subheader("Loan Information")
        
        # Loan details
        loan_intent = st.selectbox(
            "Loan Intent",
            ["EDUCATION", "PERSONAL", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        
        loan_grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )
        
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
        
        # Calculate loan percent income
        loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0
        st.info(f"Loan Percent Income: {loan_percent_income:.4f}")
        
        # Show feature summary
        st.subheader("Feature Summary")
        st.info(f"Debt-to-Income Ratio: {loan_percent_income:.2f}")
        
    # Predict button
    predict_button = st.button("Generate Features for Prediction", use_container_width=True)

with tab2:
    # This tab will show results after prediction
    if not predict_button:
        st.info("Please enter your data and click 'Generate Features for Prediction' to see results.")
    else:
        st.subheader("Prediction Preparation")
        
        # Create a dataframe with the raw input features
        input_data = {
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "person_home_ownership": person_home_ownership,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "cb_person_default_on_file": cb_person_default_on_file
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Show the raw input data
        st.subheader("Raw Input Data")
        st.dataframe(df)
        
        # Apply one-hot encoding to categorical columns
        cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        df_encoded = one_hot_encoder(df.copy(), cat_cols)
        
        # Create individual one-hot encoded variables for clarity
        st.subheader("One-Hot Encoded Variables (Individual)")
        
        # Create a dataframe to hold each one-hot encoded variable as a separate column
        ohe_data = {}
        
        # Person home ownership
        for option in ["MORTGAGE", "RENT", "OWN", "OTHER"]:
            column_name = f"person_home_ownership_{option}"
            value = 1 if person_home_ownership == option else 0
            ohe_data[column_name] = [value]
            
        # Loan intent
        for option in ["EDUCATION", "PERSONAL", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]:
            column_name = f"loan_intent_{option}"
            value = 1 if loan_intent == option else 0
            ohe_data[column_name] = [value]
            
        # Loan grade
        for option in ["A", "B", "C", "D", "E", "F", "G"]:
            column_name = f"loan_grade_{option}"
            value = 1 if loan_grade == option else 0
            ohe_data[column_name] = [value]
            
        # Default on file
        for option in ["Y", "N"]:
            column_name = f"cb_person_default_on_file_{option}"
            value = 1 if cb_person_default_on_file == option else 0
            ohe_data[column_name] = [value]
            
        # Create a dataframe with the individual one-hot encoded variables
        ohe_df = pd.DataFrame(ohe_data)
        
        # Display the one-hot encoded variables
        st.dataframe(ohe_df)
        
        # Create multi-column layout for better display
        cols = st.columns(2)
        
        # Display each categorical variable's encoding explicitly
        with cols[0]:
            st.write("### Home Ownership Variables")
            st.write(f"**person_home_ownership_MORTGAGE** = {1 if person_home_ownership == 'MORTGAGE' else 0}")
            st.write(f"**person_home_ownership_OTHER** = {1 if person_home_ownership == 'OTHER' else 0}")
            st.write(f"**person_home_ownership_OWN** = {1 if person_home_ownership == 'OWN' else 0}")
            st.write(f"**person_home_ownership_RENT** = {1 if person_home_ownership == 'RENT' else 0}")
            
            st.write("### Default on File Variables")
            st.write(f"**cb_person_default_on_file_N** = {1 if cb_person_default_on_file == 'N' else 0}")
            st.write(f"**cb_person_default_on_file_Y** = {1 if cb_person_default_on_file == 'Y' else 0}")
                
        with cols[1]:
            st.write("### Loan Intent Variables") 
            st.write(f"**loan_intent_DEBTCONSOLIDATION** = {1 if loan_intent == 'DEBTCONSOLIDATION' else 0}")
            st.write(f"**loan_intent_EDUCATION** = {1 if loan_intent == 'EDUCATION' else 0}")
            st.write(f"**loan_intent_HOMEIMPROVEMENT** = {1 if loan_intent == 'HOMEIMPROVEMENT' else 0}")
            st.write(f"**loan_intent_MEDICAL** = {1 if loan_intent == 'MEDICAL' else 0}")
            st.write(f"**loan_intent_PERSONAL** = {1 if loan_intent == 'PERSONAL' else 0}")
            st.write(f"**loan_intent_VENTURE** = {1 if loan_intent == 'VENTURE' else 0}")
                
            st.write("### Loan Grade Variables")
            st.write(f"**loan_grade_A** = {1 if loan_grade == 'A' else 0}")
            st.write(f"**loan_grade_B** = {1 if loan_grade == 'B' else 0}")
            st.write(f"**loan_grade_C** = {1 if loan_grade == 'C' else 0}")
            st.write(f"**loan_grade_D** = {1 if loan_grade == 'D' else 0}")
            st.write(f"**loan_grade_E** = {1 if loan_grade == 'E' else 0}")
            st.write(f"**loan_grade_F** = {1 if loan_grade == 'F' else 0}")
            st.write(f"**loan_grade_G** = {1 if loan_grade == 'G' else 0}")
        
        # Show final columns for model prediction
        st.markdown("---")
        st.subheader("Final Model Prediction Columns")
        
        # Combine numerical and one-hot encoded features
        final_df = pd.concat([df[["person_age", "person_income", "person_emp_length", 
                                  "loan_amnt", "loan_int_rate", "loan_percent_income", 
                                  "cb_person_cred_hist_length"]], 
                              ohe_df], axis=1)
        
        # Display all columns that will be used for prediction
        final_cols = final_df.columns.tolist()
        
        # Create a nice display of the columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Numerical Features")
            numerical_cols = ["person_age", "person_income", "person_emp_length", 
                             "loan_amnt", "loan_int_rate", "loan_percent_income", 
                             "cb_person_cred_hist_length"]
            for col in numerical_cols:
                st.write(f"• **{col}**: {final_df[col].values[0]}")
        
        with col2:
            st.markdown("#### One-Hot Encoded Features (Active)")
            encoded_cols = [col for col in final_cols if col not in numerical_cols]
            # Only show the active (value=1) encoded features to avoid clutter
            active_encoded_cols = [col for col in encoded_cols if final_df[col].values[0] == 1]
            for col in active_encoded_cols:
                st.write(f"• **{col}**: 1")
        
        # Display the full column list in an expandable section
        with st.expander("View All Model Input Features"):
            # Create three columns for better formatting
            cols = st.columns(3)
            for i, col in enumerate(final_cols):
                cols[i % 3].write(f"{i+1}. {col}")
            
            # Also provide the data as a JSON format for reference
            st.write("#### Features as JSON (for API use):")
            st.json(final_df.to_dict(orient='records')[0])
        
        # Show final prediction input dataframe
        st.subheader("Final Prediction Input Data")
        st.dataframe(final_df)
        
        # Confirm prediction step
        run_prediction = st.button("Run Prediction with These Features", use_container_width=True)
        
        if run_prediction:
            # Make prediction using the final dataframe
            prediction, probability = predict_loan_approval(final_df)
            
            # Display prediction results
            st.markdown("---")
            st.subheader("Loan Default Prediction")
            
            if prediction is not None:
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if prediction == 1:
                        st.error("Loan Default Risk: HIGH ❌")
                        st.markdown("Model predicts borrower may default on the loan")
                    else:
                        st.success("Loan Default Risk: LOW ✅")
                        st.markdown("Model predicts borrower will repay the loan")
                
                with col2:
                    if prediction == 1:
                        default_prob = probability[1] * 100
                        st.metric("Default Probability", f"{default_prob:.2f}%")
                    else:
                        repay_prob = probability[0] * 100
                        st.metric("Repayment Probability", f"{repay_prob:.2f}%")
                
                with col3:
                    st.markdown("### Key Risk Factors")
                    
                    # Simple rules to explain the decision
                    risk_factors = []
                    
                    if cb_person_default_on_file == "Y":
                        risk_factors.append("Previous default on record")
                    
                    if loan_percent_income > 0.2:
                        risk_factors.append("High debt-to-income ratio")
                    
                    if loan_grade in ["E", "F", "G"]:
                        risk_factors.append("Low credit grade")
                    
                    if person_emp_length < 2:
                        risk_factors.append("Short employment history")
                    
                    if len(risk_factors) > 0:
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("• No significant risk factors")
            
            # Feature importance visualization
            st.markdown("---")
            st.subheader("Feature Analysis")
            
            # Show input relative to typical values
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Income to Loan Amount Relationship")
                # Create a simple scatter visualization
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.scatter(x=[50000, 65000, 80000], y=[8000, 12000, 15000], alpha=0.5, label="Typical Cases")
                ax.scatter(x=[person_income], y=[loan_amnt], color='red', s=100, label="Your Application")
                ax.set_xlabel("Income ($)")
                ax.set_ylabel("Loan Amount ($)")
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Loan Grade Distribution")
                # Create a simple bar chart for loan grades
                fig, ax = plt.subplots(figsize=(5, 3))
                grades = ["A", "B", "C", "D", "E", "F", "G"]
                default_rates = [0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.60]
                
                # Highlight the selected grade
                colors = ['blue'] * 7
                grade_idx = grades.index(loan_grade)
                colors[grade_idx] = 'red'
                
                ax.bar(grades, default_rates, color=colors)
                ax.set_xlabel("Loan Grade")
                ax.set_ylabel("Default Rate")
                ax.set_title(f"Your Loan Grade: {loan_grade}")
                st.pyplot(fig)
            
            # Model information
            st.markdown("---")
            st.subheader("Model Information")
            st.markdown("""
            This prediction uses a LightGBM model trained on historical loan data. The model was saved as `LightGBM_bestmodel.pkl`.
            
            The model takes into account:
            - Personal information (age, income, employment length)
            - Loan details (amount, interest rate, purpose)
            - Credit information (credit history length, previous defaults)
            - One-hot encoded categorical variables (home ownership, loan intent, loan grade)
            
            The prediction indicates the likelihood of a loan defaulting based on these factors.
            """)
        else:
            st.info("Click 'Run Prediction with These Features' to see prediction results")

# If model not yet loaded and no prediction made
if not predict_button:
    st.markdown("---")
    st.info("""
    ### About the Loan Prediction Model
    
    This tool uses a machine learning model to predict loan default risk based on various factors.
    
    To use:
    1. Fill in the applicant information fields
    2. Click "Generate Features for Prediction" to prepare the model input
    3. Review the feature encodings and final prediction columns
    4. Click "Run Prediction with These Features" to see the prediction results
    
    The prediction is based on a LightGBM model trained on historical loan data.
    """)

# Footer
st.markdown("---")
st.markdown("Loan Prediction System | Default Risk Assessment Tool")
