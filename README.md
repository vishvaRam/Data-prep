# Data Preparation Project for RBI Circular QA Dataset

A comprehensive data preparation and processing pipeline built with Python and Docker, designed to generate synthetic Question-Answer pairs from RBI Circulars using Google's Gemini 2.0-flash model. This project encompasses the entire pipeline from web scraping RBI circulars to pushing the generated dataset to the Hugging Face Hub.


## 🎯 Project Goal
This project aims to create a high-quality Question-Answer dataset from RBI Circulars for fine-tuning Large Language Models. The dataset is available on Hugging Face Hub at [Vishva007/RBI-Circular-QA-Dataset](https://huggingface.co/datasets/Vishva007/RBI-Circular-QA-Dataset).

## 🚀 Features

- **Web Scraping**: Automated extraction of RBI Circulars from the official website
- **PDF Processing**: Tools for processing and extracting text from RBI circular PDFs into high quality Markdown files
- **AI Integration**: Uses Google's Gemini 2.0-flash model for generating synthetic QA pairs
- **Dataset Generation**: Automated pipeline for creating high-quality QA pairs
- **Hugging Face Integration**: Direct upload capability to Hugging Face Hub
- **Docker Support**: Containerized environment with GPU support
- **Test Scripts**: Included test scripts for validation
- **Utility Functions**: Reusable utility modules for common tasks

## 📋 Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Python 3.x

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/vishvaRam/Data-Prep-for-LLM-fine-tuning.git
cd Data_prep
```

2. Build and start the Docker container:
```bash
docker-compose up --build
```

## 🏗️ Project Structure

```
.
├── Code/
│   ├── AI-tasks/        # AI-related processing tasks
│   ├── Data/            # Data storage directory
│   ├── Utils/           # Utility functions
│   ├── Test-scripts/    # Testing and validation scripts
│   ├── prepare-dataset/ # Dataset preparation tools
│   ├── process-md/      # Markdown processing tools
│   ├── convert-markdown/# Markdown conversion utilities
│   ├── fetch-data/      # Data fetching utilities
│   ├── main.py          # Print the Config of the Docker Image
│   ├── Dockerfile       # Docker configuration
│   └── requirements.txt # Python dependencies
├── .devcontainer/       # Development container configuration
└── docker-compose.yml   # Docker Compose configuration
```

## 🔧 Configuration

The project uses Docker Compose for configuration. Key settings include:

- GPU support enabled
- Volume mounting for code persistence
- Network configuration for external connectivity (If Langfuse is used)

## 📦 Dependencies

Key Python packages include:
- google-generativeai (for Gemini 2.0-flash model integration)
- langchain (for LLM operations)
- langfuse (for Monitor)
- marker-pdf (for PDF to Markdown processing)
- selenium (for web scraping)
- and more (see requirements.txt for complete list)

## 🚀 Usage

1. Start the container:
```bash
docker-compose up
```

2. The project workflow:
   - Web scrape RBI circulars using the fetch-data module
   - Process PDFs using the process-md module
   - Convert the md files into chunks
   - Generate QA pairs using Gemini 2.0-flash model from the chunks
   - Validate and prepare the dataset (if necessary)
   - Push to Hugging Face Hub

## 📊 Dataset Information

The generated dataset contains synthetic Question-Answer pairs created from RBI Circulars using the Gemini 2.0-flash model. The dataset is structured to facilitate fine-tuning of Large Language Models for better understanding and processing of RBI circulars.

Dataset Location: [Vishva007/RBI-Circular-QA-Dataset](https://huggingface.co/datasets/Vishva007/RBI-Circular-QA-Dataset)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch
- OpenCV
- Langfuse
- marker-pdf
- All other open-source contributors 