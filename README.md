# Loan Prediction System - Streamlit Application

This project is a Streamlit application for analyzing loan data and predicting loan approval status.

## Structure

- `streamlit/Home.py`: Main application page
- `streamlit/pages/`
  - `Loan_Dataset.py`: Page for exploring the dataset
  - `EDA_Visualization.py`: Page for data visualization and analysis
- `src/`: Core functionality modules
  - Data import, analysis, preparation, visualization, and modeling

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run streamlit/Home.py
```

The application will open in your default web browser.

## Features

- **Home**: Overview of the application
- **Loan Dataset**: Explore the loan datasets
  - View training, testing, or combined data
  - Analyze individual features
  - View statistics and distributions
- **EDA & Visualization**: Interactive data visualization
  - Explore numerical and categorical distributions
  - Analyze correlations between features
  - Create custom plots (scatter plots, bar charts, etc.)

## Data

The application uses loan data files located in the `data/` directory:
- `train.csv`: Training dataset
- `test.csv`: Testing dataset