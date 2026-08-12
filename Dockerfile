# ──────────────────────────────────────────────────────────────
# Stage 1 — Builder
# Uses official Rust image. litert-sys build script will download
# the Linux .so files from crates.io into ~/.cache/litert-sys/
# ──────────────────────────────────────────────────────────────
FROM rust:1.89-slim-bookworm AS builder

# Build-time dependencies (build-essential & g++ provide libstdc++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    pkg-config \
    libssl-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependency layer — copy manifests first so cargo fetch is cached
# unless Cargo.toml / Cargo.lock change.
COPY Cargo.toml Cargo.lock build.rs ./

# Create a dummy lib.rs and server binary so `cargo fetch` resolves all deps
RUN mkdir -p src/server src/bin src/embedder && \
    echo 'pub fn main() {}' > src/bin/server.rs && \
    echo 'pub mod server { pub mod state {} pub mod error {} pub mod handlers {} pub mod routes {} }' > src/lib.rs && \
    touch src/embedder/mod.rs src/embedder/litert_embedder.rs src/embedder/ort_embedder.rs && \
    cargo fetch --locked

# Now copy the real source and build in release mode
COPY src ./src

RUN cargo build --release --bin faceid-server --locked && \
    # Strip RPATH from the binary; we will set LD_LIBRARY_PATH at runtime
    objcopy --strip-debug target/release/faceid-server 2>/dev/null || true

# ──────────────────────────────────────────────────────────────
# Stage 2 — Runtime image
# Minimal Debian Bookworm image: only copies the binary,
# LiteRT shared libs, and model files.
# ──────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# Runtime C++ standard library & curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy LiteRT Linux shared libraries dynamically regardless of target architecture (x86_64 vs aarch64)
COPY --from=builder /root/.cache/litert-sys/v0.10.2/ /tmp/litert-cache/
RUN mkdir -p /usr/local/lib/litert && \
    cp -rn /tmp/litert-cache/*/* /usr/local/lib/litert/ 2>/dev/null || true && \
    rm -rf /tmp/litert-cache

# The compiled server binary
COPY --from=builder /app/target/release/faceid-server /usr/local/bin/faceid-server

# Model files (baked into the image for portability; mount a volume to override)
COPY models/ /app/models/

# Persistent registry store mounted as a named volume
VOLUME ["/app/data"]

# Expose HTTP API port
EXPOSE 8080

# Make shared libs visible to the dynamic linker
ENV LD_LIBRARY_PATH=/usr/local/lib/litert

# Health check via /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

WORKDIR /app

ENTRYPOINT ["/usr/local/bin/faceid-server"]
CMD [ \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--detector", "/app/models/yunet_fp16.tflite", \
    "--antispoof", "/app/models/silentface.tflite", \
    "--embedder", "/app/models/facenet512.tflite", \
    "--registry", "/app/data/registry.json" \
]
