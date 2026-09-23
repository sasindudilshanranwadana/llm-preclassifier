FROM python:3.11-slim@sha256:174bec68e0451bffabbb08c7d5d21c6b253f772d81d52b9558af97bb3159b761 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app --home /nonexistent app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

FROM base AS runtime
USER app
EXPOSE 8802
CMD ["uvicorn", "--factory", "llm_preclassifier.api:create_app", "--host", "0.0.0.0", "--port", "8802", "--no-access-log"]

# Runtime with the offline semantic safety layer; the model is baked in so the
# container never fetches weights at runtime.
FROM base AS runtime-semantic
ENV SEMANTIC_MODEL=BAAI/bge-small-en-v1.5 \
    SEMANTIC_CACHE_DIR=/opt/models \
    HF_HUB_OFFLINE=1
RUN pip install --no-cache-dir ".[semantic]" \
    && HF_HUB_OFFLINE=0 python -c "from fastembed import TextEmbedding; TextEmbedding('$SEMANTIC_MODEL', cache_dir='$SEMANTIC_CACHE_DIR')" \
    && chmod -R a+rX /opt/models
USER app
EXPOSE 8802
CMD ["uvicorn", "--factory", "llm_preclassifier.api:create_app", "--host", "0.0.0.0", "--port", "8802", "--no-access-log"]
