FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git curl \
    cmake build-essential \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
    patchelf libavdevice-dev libavfilter-dev libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.mujoco \
    && curl -SL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    | tar -xvzf - -C /root/.mujoco

ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}"
ENV MUJOCO_PY_MUJOCO_PATH="/root/.mujoco/mujoco210"
ENV PYOPENGL_PLATFORM="osmesa"

WORKDIR /workspace
COPY requirements_docker.txt .

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install mujoco-py==2.1.2.14
RUN pip3 install -r requirements_docker.txt
