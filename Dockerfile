# Aggregating Inertia: the computations of the paper in one image.
#   docker build -t aggregating-inertia .
#   docker run --rm -v "$PWD/output:/work/output" -v "$PWD/figures:/work/figures" aggregating-inertia          # everything
#   docker run --rm -v "$PWD/figures:/work/figures" aggregating-inertia figures                                 # figures only
# Stage 1 builds the Rust engine; stage 2 adds Julia with the plotting environment precompiled.

FROM rust:1.95 AS engine
WORKDIR /src
COPY rust/Cargo.toml rust/Cargo.lock rust/
COPY rust/src rust/src
COPY rust/examples rust/examples
RUN cargo build --release --locked --manifest-path rust/Cargo.toml \
 && cargo test --release --locked --manifest-path rust/Cargo.toml --lib

FROM julia:1.12
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*
WORKDIR /work
ENV JULIA_PROJECT=/work/julia JULIA_NUM_THREADS=auto
COPY julia/Project.toml julia/Manifest.toml julia/
RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
COPY julia/ julia/
COPY --from=engine /src/rust/target/release/inertia /usr/local/bin/inertia
COPY Makefile ./
COPY output/ output/
ENTRYPOINT ["make", "ENGINE=/usr/local/bin/inertia"]
CMD ["all"]
