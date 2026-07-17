# Dockerfile — builds the container image for the API.
# Layered so a code edit doesn't re-download packages or the browser:
#   1. deps      (rebuilds only when pyproject.toml/uv.lock change)
#   2. Chromium  (rebuilds only when the deps layer does)
#   3. app code  (changes often, cheap)
# Python 3.13-slim matches the dev environment the uv.lock was resolved on.

FROM python:3.13-slim

# Install uv by copying the static binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /code

# 1. Dependencies — cached until pyproject.toml/uv.lock actually change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Playwright headless Chromium + its OS libraries, so JS-rendered sites
#    (Framer/Webflow/SPA) are readable INSIDE the container exactly as in dev.
#    Without this the render fallback fails and JS sites crawl thin — the very
#    bug Phase 6 fixed would silently return in production. --with-deps runs the
#    apt-get install of the shared libraries Chromium needs.
RUN uv run --no-sync playwright install --with-deps chromium

# 3. Application code (includes app/static — the dashboard is served from there).
COPY app/ ./app/

EXPOSE 8000

# --host 0.0.0.0 makes it reachable from outside the container.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
