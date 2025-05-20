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
RUN wget --no-check-certificate https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt install ./google-chrome-stable_current_amd64.deb -y --no-install-recommends && \
    rm ./google-chrome-stable_current_amd64.deb

RUN python -m pip install --upgrade pip
    
# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

ENV GEMINI_API_KEY=AIzaSyDbLuV1V0QyI1qI-OviHaaA3LuqdjcSGzY
ENV LANGSMITH_TRACING=true
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
ENV LANGSMITH_API_KEY="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"
ENV LANGSMITH_PROJECT="data-prep"

# Expose the port FastAPI will run on
EXPOSE 8888

# Set the default command
CMD ["python", "main.py"]