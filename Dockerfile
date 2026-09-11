# ==========================================
# Stage 1: Build Wheels
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir build setuptools>=61.0

COPY pyproject.toml README.md ./
COPY apollo_sim/ ./apollo_sim/
COPY lunar_lander/ ./lunar_lander/

RUN python -m build --wheel

# ==========================================
# Stage 2: Runtime Image
# ==========================================
FROM python:3.12-slim AS runner

# Create unprivileged apollo user (UID 10001)
RUN groupadd -g 10001 apollo && \
    useradd -u 10001 -g apollo -s /bin/bash -m apollo

WORKDIR /app

# Install wheels
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl pytest pytest-cov && rm -f /tmp/*.whl

# Copy simulation modules & tests for container smoke verification
COPY --chown=apollo:apollo apollo_sim/ /app/apollo_sim/
COPY --chown=apollo:apollo lunar_lander/ /app/lunar_lander/
COPY --chown=apollo:apollo pyproject.toml /app/

USER apollo

# Default entrypoint runs full Apollo 11 simulation
ENTRYPOINT ["apollo-sim"]
CMD ["--mission", "apollo11"]
