# Use a PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.12-py3

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

RUN python -m pip install --upgrade pip
    
# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8888

# Set the default command
CMD ["python", "main.py"]