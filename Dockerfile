# Dockerfile — builds the container image for the API.
# Two-stage layout: dependencies are installed in a layer that only rebuilds
# when pyproject.toml/uv.lock change, so code edits don't re-download packages.

FROM python:3.12-slim

# Install uv by copying the static binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /code

# 1. Copy only the dependency manifests and install — this layer is cached
#    until dependencies actually change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Copy the application code (changes often, cheap to rebuild).
COPY app/ ./app/

# Run the server. --host 0.0.0.0 makes it reachable from outside the container.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
