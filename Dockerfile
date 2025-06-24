FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Install system packages
RUN apt-get update && apt-get install -y \
    tzdata wget build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev \
    libexpat1-dev liblzma-dev tk-dev libffi-dev \
    curl git ffmpeg libsm6 libxext6 cmake && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.11.9
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    tar xvf Python-3.11.9.tgz && cd Python-3.11.9 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-3.11.9*

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Fix symlinks
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /workspaces

# Copy project
COPY . .

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root

# Install torch with CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone and build GroundingDINO (needed for compiled ops)
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn && \
    sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu && \
    sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' ms_deform_attn_cuda.cu && \
    cd /workspaces/GroundingDINO && \
    pip install -q -e . && \
    python setup.py build_ext --inplace

# Install Segment Anything
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Install remaining utilities
RUN pip install opencv-python transformers

# Create persistent volume directory
VOLUME /weights

# Run weight download script (uses /weights)
RUN poetry lock && \
    poetry install && \
    poetry run python get_weights.py

# Apply patch to grounding_dino, that should have worked as part of the clone
RUN chmod +x dino_fix.sh

RUN ./dino_fix.sh
