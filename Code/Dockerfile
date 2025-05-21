# Use a PyTorch base image
FROM vishva123/nvdia-cuda-12.6.2-cudnn-ubuntu24.04-py-3.10:latest

# Set the working directory
WORKDIR /app

# Update package lists and install system dependencies first
# Running apt-get update and install commands in a single RUN layer
# to reduce image size and improve caching.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgtk-3-0 \
    libnss3 \
    libxss1 \
    wget \
    gnupg \
    ca-certificates \
    software-properties-common \
    apt-transport-https && \
    rm -rf /var/lib/apt/lists/*

# --- Install Google Chrome with its official repository ---
# 1. Download and add Google's public signing key
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    # 2. Add Google Chrome's stable repository to apt sources
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list && \
    # 3. Update apt package lists again to include the new repository
    apt-get update && \
    # 4. Install google-chrome-stable using apt (which will now resolve dependencies)
    apt-get install -y --no-install-recommends google-chrome-stable && \
    # 5. Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# --- UV Installation and Setup ---

# Install uv globally.
# uv is installed to /root/.cargo/bin by default when run as root.
# Install uv and add it to PATH permanently
# Install uv and add it to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh  | sh && \
    cp $HOME/.local/bin/uv /usr/local/bin/uv && \
    cp $HOME/.local/bin/uvx /usr/local/bin/uvx

# Add uv to PATH (optional, but safe)
ENV PATH="/usr/local/bin:${PATH}"

# Now uv will be found
RUN uv --version

# Copy pyproject.toml and uv.lock first to leverage Docker's caching.
COPY pyproject.toml .
COPY uv.lock .

# Install Python dependencies using uv.
# uv sync ensures the environment matches pyproject.toml and uv.lock.
RUN uv sync

# --- End UV Setup ---

# Copy the rest of the application code
COPY . .

# Define a default command to run when the container starts
CMD ["uv", "run", "main.py"]