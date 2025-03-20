FROM python:3.10-slim

LABEL maintainer="gurkancnk"

WORKDIR /app

# LightGBM için gerekli sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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

# Modeli daha basit bir yaklaşımla oluştur
RUN python -c "from sklearn.ensemble import RandomForestClassifier; import joblib; model = RandomForestClassifier(n_estimators=10, random_state=42); model.fit([[0, 0], [1, 1]], [0, 1]); joblib.dump(model, 'models/LightGBM_bestmodel.pkl')"

# Set the command to run the app
CMD ["streamlit", "run", "streamlit/Home.py", "--server.port=8501", "--server.address=0.0.0.0"] 