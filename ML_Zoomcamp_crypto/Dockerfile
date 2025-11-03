# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and lock file first for dependency caching
COPY Pipfile Pipfile.lock* ./

# Install dependencies system-wide
RUN pipenv install --system --deploy

# Copy app files
COPY predict-service.py .
COPY good_bad_trader_log_reg.bin .

# Expose FastAPI port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "predict-service:app", "--host", "0.0.0.0", "--port", "8000"]
