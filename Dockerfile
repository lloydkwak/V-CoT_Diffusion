# ==============================================================================
# V-CoT Diffusion Policy - Hybrid Dockerfile
# ==============================================================================
# Retains the legacy environment (CUDA 11.8, Python 3.8) while resolving 
# EGL rendering errors and Gym/NumPy version conflicts for V-CoT training.

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# ------------------------------------------------------------------------------
# 1. System Packages & OSMesa (Headless Rendering)
# ------------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git curl \
    cmake build-essential \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
    patchelf libavdevice-dev libavfilter-dev libzmq3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 2. Legacy MuJoCo 210 (Backward Compatibility)
# ------------------------------------------------------------------------------
RUN mkdir -p /root/.mujoco \
    && curl -SL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    | tar -xvzf - -C /root/.mujoco

ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}"
ENV MUJOCO_PY_MUJOCO_PATH="/root/.mujoco/mujoco210"

# ------------------------------------------------------------------------------
# 3. [CRITICAL FIX] Force CPU Rendering (OSMesa)
# ------------------------------------------------------------------------------
# Prevents GPU resource allocation errors (EGL_BAD_ALLOC) during 
# multi-process evaluation (AsyncVectorEnv).
ENV PYOPENGL_PLATFORM="osmesa"
ENV MUJOCO_GL="osmesa"
ENV PYTHONPATH="/workspace/diffusion_policy:$PYTHONPATH"

# ------------------------------------------------------------------------------
# 4. Workspace Setup & Dependencies
# ------------------------------------------------------------------------------
WORKDIR /workspace
COPY requirements.txt .  # <-- 여기 수정

RUN pip3 install --upgrade pip setuptools==65.5.0 wheel
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Legacy Python bindings for MuJoCo
RUN pip3 install mujoco-py==2.1.2.14

# Main requirements
RUN pip3 install -r requirements.txt  # <-- 여기 수정

# ------------------------------------------------------------------------------
# 5. [CRITICAL FIX] Resolve Dependency Conflicts (Pin Versions)
# ------------------------------------------------------------------------------
RUN pip3 install "numpy==1.23.5" "gym==0.24.1" "mujoco>=3.1.0"

CMD ["/bin/bash"]
