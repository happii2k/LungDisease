GitHub README
text
# 🫁 Lung Disease X-ray Classification

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![Azure](https://img.shields.io/badge/Azure-Deployed-0078D4?style=for-the-badge&logo=microsoft-azure)

An end-to-end deep learning pipeline for classifying chest X-ray images into **Normal** and **Pneumonia** categories using a custom-built Convolutional Neural Network (CNN). The project features a production-ready FastAPI backend, Docker containerization, and cloud deployment on Microsoft Azure.

## 🎯 Project Overview

This project implements a complete MLOps pipeline for medical image classification:

- **Custom CNN Architecture**: Built from scratch without pretrained models to understand CNN fundamentals
- **Binary Classification**: Distinguishes between Normal and Pneumonia chest X-rays
- **Production API**: FastAPI-based REST endpoints for real-time predictions
- **Cloud Deployment**: Containerized and deployed on Azure Container Instances
- **Modular Architecture**: Clean, maintainable codebase following software engineering best practices

## 🏗️ Architecture

┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Data Layer │────▶│ Model Layer │────▶│ API Layer │
│ (Ingestion & │ │ (Training & │ │ (FastAPI & │
│ Preprocessing)│ │ Evaluation) │ │ Inference) │
└─────────────────┘ └──────────────────┘ └─────────────────┘
│ │ │
▼ ▼ ▼
┌─────────────────────────────────────────────────────────────────┐
│ Azure Cloud Deployment │
│ (Azure Container Registry → Azure Container Instances) │
└─────────────────────────────────────────────────────────────────┘

text

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, torchvision, Custom CNN |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Containerization** | Docker, Docker Compose |
| **Cloud** | Azure Container Registry, Azure Container Instances |
| **Data Processing** | PIL, NumPy, OpenCV |
| **Development** | Python 3.12, Git, VS Code |

## 📁 Project Structure

LungDisease/
├── src/
│ ├── components/ # Data ingestion, transformation, training
│ ├── pipeline/ # Training and prediction pipelines
│ ├── dl/ # Deep learning models (CustomNN)
│ ├── utils/ # Utility functions
│ ├── exception.py # Custom exception handling
│ ├── logger.py # Logging configuration
│ └── constant/ # Project constants
├── artifacts/ # Model artifacts and processed data
├── static/ # Frontend assets (HTML, CSS, JS)
├── templates/ # Jinja2 templates
├── app.py # FastAPI application
├── Dockerfile # Container configuration
├── requirements.txt # Python dependencies
└── README.md

text

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker Desktop (optional for containerization)
- Azure CLI (optional for cloud deployment)

### Local Installation

1. **Clone the repository**
git clone https://github.com/happii2k/LungDisease.git
cd LungDisease

text

2. **Create virtual environment**
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

text

3. **Install dependencies**
pip install -r requirements.txt

text

4. **Run training pipeline**
python -m src.pipeline.training_pipeline

text

5. **Start the API server**
python app.py

text

6. **Access the application**
- API Docs: `http://localhost:8000/docs`
- Web Interface: `http://localhost:8000`

### Docker Deployment

Build the image
docker build -t lung-disease-classifier .

Run the container
docker run -p 8000:8000 lung-disease-classifier

text

### Azure Deployment

Login to Azure
az login

Create container registry
az acr create --resource-group <your-rg> --name lungdisease --sku Basic

Build and push image
az acr build --registry lungdisease --image xray-classifier:latest .

Deploy to Azure Container Instances
az container create
--resource-group <your-rg>
--name xray-classifier
--image lungdisease.azurecr.io/xray-classifier:latest
--ports 8000
--dns-name-label xray-classifier-app

text

## 📊 Model Architecture

The custom CNN architecture consists of:

Net(
(conv1): Conv2d(3, 32, kernel_size=3, padding=1)
(conv2): Conv2d(32, 64, kernel_size=3, padding=1)
(conv3): Conv2d(64, 128, kernel_size=3, padding=1)
(pool): MaxPool2d(kernel_size=2, stride=2)
(fc1): Linear(128 * 28 * 28, 512)
(fc2): Linear(512, 2)
(dropout): Dropout(0.5)
)

text

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface for image upload |
| `GET` | `/health` | Health check endpoint |
| `POST` | `/predict` | Classify uploaded X-ray image |

### Example API Request

curl -X POST "http://localhost:8000/predict"
-H "Content-Type: multipart/form-data"
-F "file=@chest_xray.jpg"

text

### Response
{
"prediction": "Pneumonia",
"confidence": 0.94
}

text

## 📈 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
- **Training Set**: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Validation Set**: 16 images
- **Test Set**: 624 images
- **Image Size**: Resized to 224×224 pixels

## 🧪 Data Augmentation

Applied transformations for robust training:
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness, contrast)
- Normalization (ImageNet statistics)

## 📝 Logging & Monitoring

Comprehensive logging implemented throughout the pipeline:
- Training progress and loss metrics
- Inference timing and predictions
- Error tracking with custom exception handling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Harsh Parihar**
- GitHub: [@happii2k](https://github.com/happii2k)
- Portfolio: [harshparihar.github.io](https://happii2k.github.io/harshparihar.github.io/)

## 🙏 Acknowledgments

- Chest X-Ray dataset from Kaggle
- PyTorch team for the deep learning framework
- FastAPI for the modern Python web framework
📄 Resume Description
Short Version (For Resume Bullet Points)
Lung Disease X-ray Classification | PyTorch, FastAPI, Docker, Azure

Developed an end-to-end deep learning pipeline to classify chest X-rays into Normal and Pneumonia categories using a custom-built CNN architecture without pretrained models

Built a production-ready REST API with FastAPI featuring real-time inference, comprehensive error handling, and logging

Containerized the application using Docker and deployed on Microsoft Azure (ACR + Container Instances), demonstrating full MLOps lifecycle expertise

Implemented modular code architecture with separate components for data ingestion, model training, and prediction pipelines

Long Version (For Resume Project Section or Cover Letter)
Lung Disease X-ray Classification System

Built a complete deep learning solution for medical image classification that identifies Pneumonia from chest X-ray images. The project showcases expertise in custom CNN development, production API design, and cloud deployment:

