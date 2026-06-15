ARG L4T_VERSION=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]

WORKDIR /workspaces/ml

# Foundational packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    curl \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Install specific versions of PyTorch, torchvision, and torchaudio
# Based on https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
RUN cd /tmp && \
    wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl \
    -O torch-2.3.0-cp310-cp310-linux_aarch64.whl && \
    wget https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl \
    -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl && \
    wget https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl \
    -O torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl && \
    pip install \
    torch-2.3.0-cp310-cp310-linux_aarch64.whl \
    torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl \
    torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl \
    && rm -f /tmp/*.whl

# Install additional packages
RUN apt-get update && apt-get install -y \
    libopenmpi3 \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    Pillow \
    ultralytics==8.3.2 \
    # PyTorch 2.3 is not supported after transformers 4.45
    "transformers<4.45" \
    dvc \
    dvc[s3] \
    python-dotenv \
    labelbox \
    labelbox[data] \
    ndjson \
    natsort \
    opencv-python \
    albumentations \
    torchmetrics \
    tqdm

# Install ONNX + ONNX Runtime
# Based on https://elinux.org/Jetson_Zoo#ONNX_Runtime
RUN cd /tmp && \
    wget https://nvidia.box.com/shared/static/6l0u97rj80ifwkk8rqbzj1try89fk26z.whl \
    -O onnxruntime-1.19.0-cp310-cp310-linux_aarch64.whl && \
    pip install --no-cache-dir \
    onnx \
    onnxruntime-1.19.0-cp310-cp310-linux_aarch64.whl \
    && rm -f /tmp/*.whl

# Install ml_utils. Note that it is a local package and thus needs to be installed after
# it has been copied.
COPY ml_utils ml_utils
RUN pip install --no-cache-dir -e ml_utils