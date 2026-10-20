# Container image for hosting ODS on any Docker host.
FROM python:3.12-slim

# libgl1/libglib2.0: OpenCV runtime libraries that the slim image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user. The app directory is created (and owned) up front because the legacy
# Docker builder (Ubuntu's docker.io) otherwise creates WORKDIR as root, and the installs below fail.
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown user:user /home/user/app
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    YOLO_CONFIG_DIR=/tmp
WORKDIR /home/user/app

# CPU-only PyTorch first: the default Linux wheels pull ~3 GB of CUDA libraries a CPU host can't use.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencies first, against a stub package, so code edits don't re-download them. The install is
# editable: ods/ stays in the app directory, which is where models.yaml, static/ and weights/ are looked up
# (ods.registry.ROOT is derived from the package location).
COPY --chown=user pyproject.toml README.md ./
RUN mkdir ods && touch ods/__init__.py && pip install --no-cache-dir -e .
COPY --chown=user ods ./ods

COPY --chown=user models.yaml ./
COPY --chown=user static ./static
COPY --chown=user weights ./weights
COPY --chown=user yolo11m.pt ./

EXPOSE 7860
CMD ["python", "-m", "ods.server"]
