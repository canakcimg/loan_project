FROM python:3.10-slim

LABEL maintainer="gurkancnk"

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Make port 8501 available for the app
EXPOSE 8501

# Create directory for model
RUN mkdir -p models

# Download default model if needed
# This is just a placeholder; you would typically have a real model
RUN echo "Creating a placeholder model file for demonstration" && \
    python -c "import joblib; import lightgbm as lgb; \
    model = lgb.LGBMClassifier(random_state=42); \
    model.fit([[0, 0], [1, 1]], [0, 1]); \
    joblib.dump(model, 'models/LightGBM_bestmodel.pkl')"

# Set the command to run the app
CMD ["streamlit", "run", "streamlit/Home.py", "--server.port=8501", "--server.address=0.0.0.0"] 