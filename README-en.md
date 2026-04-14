[🇧🇷 Leia em Português](README-pt.md)

# Corn Leaf Disease Classifier using Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![React](https://img.shields.io/badge/React-18.x-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.x-009688?style=for-the-badge&logo=fastapi&logoColor=white)

<p align="center">
  <em>A comparative analysis of CNNs for precision diagnosis in agriculture, with a web application proof of concept.</em>
</p>
<p align="center">
  Undergraduate Thesis by <strong>Gustavo Kermaunar Volobueff</strong>
</p>

---

### 📋 Table of Contents

- [📖 About the Project](#-about-the-project)
- [📸 Application Demonstration](#-application-demonstration)
- [🛠️ Technologies Used](#️-technologies-used)
- [🏆 Model Results](#-model-results)
- [📂 Repository Structure](#-repository-structure)
- [🚀 How to Run the Project](#-how-to-run-the-project)
- [📜 License](#-license)
- [👨‍💻 Author](#-author)

---

## 📖 About the Project

Foliar diseases in highly important agricultural crops, such as corn, represent a critical challenge that negatively impacts productivity, the agribusiness economy, and food security. Traditional diagnosis, based on visual inspection, is often subjective and inefficient on a large scale, creating a need for automatic and accurate tools. To address this gap, this work proposes the development and validation of a high-performance classification system for corn leaf pathologies using deep learning and computer vision techniques.

To achieve this, three Convolutional Neural Network (CNN) architectures (ResNet101, MobileNetV3-Large, and EfficientNetV2M) were compared. They were trained on two public datasets (PlantVillage + PlantDoc and CD\&S) through a robust preprocessing pipeline, featuring class balancing and the application of regularization techniques.

---

## 📸 Application Demonstration

The viability of the solution was demonstrated through an interactive web application that allows the user to submit an image of a corn leaf and receive its classification in real time.

| Home Screen | Classification Result |
| :---: | :---: |
| ![Application Home Screen](.github/assets/page1.png) | ![Result Screen](.github/assets/page2.png) |
| *The user can drag or click to select an image.* | *The model classifies the disease and displays the prediction confidence.* |

---

## 🛠️ Technologies Used

The project was divided into three main components, each with its own set of technologies:

- **🤖 Artificial Intelligence / Machine Learning:**
  - **Python:** Main language for training and the API.
  - **TensorFlow/Keras:** Framework for building and training the CNN models.
  - **Scikit-learn:** Used to generate evaluation metrics (Confusion Matrix, Classification Report).
  - **NumPy & Matplotlib:** For numerical data manipulation and visualization.

- **⚙️ Backend (API):**
  - **FastAPI:** High-performance Python framework for building the API that serves the model.

- **🖥️ Frontend:**
  - **React (with Vite):** JavaScript library for building the user interface.
  - **TailwindCSS:** CSS framework for fast and responsive styling.
  - **Axios:** HTTP client for communication between the frontend and the API.

---

## 🏆 Model Results

The comparative analysis among the three architectures revealed the superiority of the **EfficientNetV2M** model, especially when trained with higher resolution images.

| Model | Best Performing Dataset | Resolution | Validation Accuracy |
| :--- | :--- | :---: | :---: |
| 🥇 **EfficientNetV2M** | **CD&S** | **480x480** | **99.72%** |
| 🥈 **EfficientNetV2M** | PlantDoc + PlantVillage | 480x480 | 98.79% |

This performance surpasses recent benchmarks in the literature, validating the effectiveness of the proposed training and regularization pipeline.

---

## 🚀 How to Run the Project

Follow the steps below to run the complete application in your local environment.

### **Important: Model Files**

> **Note:** The trained model files (`.keras`) are not included in this repository due to their excessive size (over 300 MB), which exceeds GitHub LFS limits. To run the API, you will need to download the model file (`efficientnetv2m_..._classifier.keras`) separately from a personal drive and place it in a folder named `saved_models` at the project root. The code in `api/main.py` expects to find the model in this location.

### Prerequisites

- **Git:** To clone the repository.
- **Python:** Version `3.12.6` (It must be exactly this version due to TensorFlow conflicts).
- **Node.js:** Version `v20.18.1` or higher (includes `npm` `10.8.2` or higher).

### 1. Clone the Repository

```bash
git clone [https://github.com/gustavokv/disease-detection.git](https://github.com/gustavokv/disease-detection.git)
cd disease-detection
```

### 2. Run the Backend (API)
Open a terminal in the project's root folder.

```bash
# Navigate to the API folder
cd api

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
.\venv\Scripts\activate  # On Windows

# Install the backend dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```
The FastAPI server will be running at http://127.0.0.1:8000.

### 3. Run the Frontend
Open a new terminal in the project's root folder.

```bash

# Navigate to the frontend folder
cd frontend

# Install the frontend dependencies
npm install

# Start the development application
npm run dev

```

The React application will be accessible in your browser at http://localhost:5173.

## 📜 License

This project is licensed under the MIT License. See the LICENSE [LICENSE](LICENSE) file for more details.

## 👨‍💻 Author
<b>Gustavo Kermaunar Volobueff</b>

[GitHub](https://github.com/gustavokv)

[LinkedIn](https://www.linkedin.com/in/gustavo-kermaunar-volobueff)
