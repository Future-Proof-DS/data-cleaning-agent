# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Keep Python lean and logs visible
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Work directory inside the image
WORKDIR /app

# Install Python dependencies first, with the exact versions in poetry.lock
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip && pip install "poetry>=2.0,<3.0" \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root

# Copy the rest of your code
COPY . .

# Streamlit listens on 8501 locally. Render passes its own port in $PORT
EXPOSE 8501
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
