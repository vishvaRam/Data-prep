FROM vishva123/nvdia-cuda-12.6-cudnn-ubuntu24.04-py-3.10-uv

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgtk-3-0 \
    libnss3 \
    libxss1 \
    wget \
    gnupg \
    python3-opencv \
    libgl1 \
    libgl1-mesa-dev \
    libgl1-mesa-dri \
    ca-certificates \
    software-properties-common \
    apt-transport-https && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub  | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/Mesa:${LD_LIBRARY_PATH}

COPY requirements.txt .

RUN uv pip install -r requirements.txt --system

COPY . .

CMD ["python", "main.py"]