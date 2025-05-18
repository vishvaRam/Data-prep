# Use a PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set the working directory
WORKDIR /workdir

# Install Python dependencies
COPY . .

# Update package lists
RUN apt-get update

RUN apt-get install -y git

# Necessary dependencies for Chrome
RUN apt install -y libgtk-3-0 libnss3 libxss1

# Install necessary dependencies for Chrome
RUN apt-get install -y wget gnupg ca-certificates

# Download and install Google Chrome
RUN wget --no-check-certificate https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt install ./google-chrome-stable_current_amd64.deb -y --no-install-recommends
RUN rm ./google-chrome-stable_current_amd64.deb

RUN pip install -r requirements.txt

ENV GEMINI_API_KEY=AIzaSyDbLuV1V0QyI1qI-OviHaaA3LuqdjcSGzY

# Expose the port FastAPI will run on
EXPOSE 8888

# Set the default command
CMD ["python", "main.py"]