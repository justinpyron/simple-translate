FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
# TODO: Build requirements.txt from pyproject.toml inside GitHub Actions workflow
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy only the necessary application files
COPY app.py interfaces.py ./

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run Streamlit with Cloud Run-compatible settings
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
