# Use a PyTorch base image
FROM vishva123/nvdia-cuda-12.6.2-cudnn-ubuntu24.04-py-3.10:latest

# Set the working directory
WORKDIR /workdir

# Update package lists and install system dependencies first
RUN apt-get update && \
    apt-get install -y \
    git \
    libgtk-3-0 \
    libnss3 \
    libxss1 \
    wget \
    gnupg \
    ca-certificates

# Download and install Google Chrome
RUN wget --no-check-certificate https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb   && \
    apt install ./google-chrome-stable_current_amd64.deb -y --no-install-recommends && \
    rm ./google-chrome-stable_current_amd64.deb

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
    
# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

