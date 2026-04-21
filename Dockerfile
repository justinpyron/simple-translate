FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
# NOTE: requirements.txt is generated during the CI/CD workflow from pyproject.toml.
# This keeps dependencies in sync without manual duplication.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary application files
COPY app.py schemas.py ./

# Port must match PORT in app.py
EXPOSE 8080

CMD ["python", "app.py"]
