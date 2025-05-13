# Use a PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Set the working directory
WORKDIR /workdir

# Install Python dependencies
COPY . .

# RUN  apt-get update && apt-get install -y \
#     git \
#     curl \
#     python3-dev \
#     build-essential \
#     libssl-dev \
#     libffi-dev \
#     libpython3-dev \
#     && rm -rf /var/lib/apt/lists/*  && \
#     pip install --upgrade pip 


# Update package lists
RUN apt-get update

RUN apt-get install git

# Necessary dependencies for Chrome
RUN apt install libgtk-3-0 libnss3 libxss1 libasound2


# Install necessary dependencies for Chrome
RUN apt-get install -y wget gnupg ca-certificates

# Download and install Google Chrome
RUN wget --no-check-certificate https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt install ./google-chrome-stable_current_amd64.deb -y --no-install-recommends
RUN rm ./google-chrome-stable_current_amd64.deb

RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8888

# Set the default command
CMD ["python", "main.py"]