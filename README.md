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

## Docker Support

### Running with Docker

You can run the application using Docker:

```bash
docker pull gurkancnk/loanproject:latest
docker run -p 8501:8501 gurkancnk/loanproject:latest
```

Then access the application at http://localhost:8501

### Using Docker Compose

Alternatively, you can use Docker Compose to run the application:

```bash
docker-compose up -d
```

### Building the Docker Image Locally

If you want to build the Docker image locally:

```bash
docker build -t gurkancnk/loanproject:latest .
```

### GitHub Actions

This project includes GitHub Actions workflows that automatically build and push the Docker image to Docker Hub when changes are pushed to the main branch.

To use this feature, you need to set the following secrets in your GitHub repository:
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token