# ============================================================================
# Stage 1: Build Rust extension wheel
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install system build dependencies and Rust toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Install build tools
RUN pip install --no-cache-dir maturin

# Copy source code and build files
COPY Cargo.toml Cargo.lock pyproject.toml README.md ./
COPY src/ ./src/
COPY pulsar/ ./pulsar/

# Build standalone wheel
RUN maturin build --release --out dist

# ============================================================================
# Stage 2: Minimal production runtime image
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 pulsar

# Copy wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/wheels/

# Install built pulsar wheel along with fastmcp for HTTP/SSE transport
RUN pip install --no-cache-dir /tmp/wheels/*.whl "fastmcp>=2.0" && \
    rm -rf /tmp/wheels

USER pulsar

# Default environment variables
ENV PULSAR_MCP_TRANSPORT=sse \
    PULSAR_MCP_HOST=0.0.0.0 \
    PULSAR_MCP_PORT=8000 \
    PULSAR_MCP_ALLOWED_HOSTS=*

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/sse')" || exit 1

ENTRYPOINT ["pulsar-mcp"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
