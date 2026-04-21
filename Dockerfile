FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
# NOTE: requirements.txt is generated during the CI/CD workflow from pyproject.toml.
# This keeps dependencies in sync without manual duplication.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary application files
COPY app.py schemas.py ./

# Cloud Run provides PORT; default matches typical Cloud Run / local expectations
ENV PORT=8080
EXPOSE 8080

# Dash app (see app.py); listen on 0.0.0.0 for container networking
CMD ["python", "-c", "import os; from app import app as dash_app; dash_app.run(host=\"0.0.0.0\", port=int(os.environ.get(\"PORT\", \"8080\")), debug=False)"]
