# 基础镜像：NVIDIA CUDA 12.8 + Ubuntu 24.04
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive

# apt 代理
RUN cat <<EOF >/etc/apt/apt.conf.d/99proxy
Acquire::http::Proxy "http://172.19.135.130:5000";
Acquire::https::Proxy "http://172.19.135.130:5000";
EOF

# 安装依赖：Python3、curl（用于安装 uv）、其他常用工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    vim \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（官方推荐的一键安装脚本）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 设置工作目录
WORKDIR /root/workspace/dr/rllm

# 容器启动时默认进入 bash 终端
CMD ["bash"]
