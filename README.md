# Parkinson's Disease Detection using Voice Features 🗣️🧠

This project implements a non-invasive system for the early and accurate detection of Parkinson's Disease by analyzing specific voice features using advanced deep learning techniques.

## 🌟 Project Overview

Parkinson's Disease is a degenerative movement disorder with profound effects on both motor and non-motor functions. Early and correct diagnosis is highly desirable for effective disease management. This system aims to fill the gap between classical diagnostic methods and advanced technology, providing a scalable, efficient, and accessible mechanism for detection.

We leverage sophisticated machine learning models, specifically recurrent neural networks (RNNs) like LSTM and GRU, to process and learn from voice characteristics without requiring invasive procedures.

## ✨ Key Features

* **Non-Invasive Diagnosis:** Utilizes voice recordings to extract features, avoiding invasive techniques.
* **Deep Learning Models:** Employs single LSTM, single GRU, and a **Hybrid LSTM-GRU** model to capture complex sequential information from voice data.
* **Feature Extraction:** Processes crucial voice features such as Jitter (variability in pitch), Shimmer (variability in amplitude), and H/N Ratio (Harmonics-to-Noise ratio).
* **Performance Comparison:** Evaluates models based on Accuracy, Precision, Recall, and F1-Score, demonstrating the superior performance of the hybrid approach.
* **User-Friendly Interface:** Built with Streamlit for easy interaction and visualization of results.

## 🛠️ Technologies Used

* **Python 3.11.x**
* [**Streamlit**](https://streamlit.io/) - For building the interactive web application.
* [**TensorFlow / Keras**](https://www.tensorflow.org/keras) - For building and training the deep learning models (LSTM, GRU).
* [**Pandas**](https://pandas.pydata.org/) - For data manipulation and analysis.
* [**NumPy**](https://numpy.org/) - For numerical operations.
* [**Scikit-learn**](https://scikit-learn.org/stable/) - For data preprocessing and evaluation metrics.
* [**Matplotlib**](https://matplotlib.org/) - For data visualization.
* [**Seaborn**](https://seaborn.pydata.org/) - For enhanced data visualization.

## 📂 Data

The project utilizes a dataset named `parkinsons_1.csv`, which typically contains various voice-related features derived from speech recordings of individuals with and without Parkinson's disease. This type of dataset is commonly sourced from public repositories like the UCI Machine Learning Repository (e.g., the Parkinson's Speech Dataset).

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.11.x installed on your system. **(It's highly recommended to install Python from [python.org](https://www.python.org/downloads/) and ensure "Add python.exe to PATH" is checked during installation to avoid common issues with Microsoft Store Python installations.)**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/your-repo-name.git](https://github.com/YourGitHubUsername/your-repo-name.git)
    cd "your-repo-name" # Navigate into your project directory
    ```
    (Replace `YourGitHubUsername` and `your-repo-name` with your actual GitHub details.)

2.  **Create a Virtual Environment:**
    It's best practice to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows (PowerShell):**
        ```powershell
        Set-ExecutionPolicy Bypass -Scope Process # Only if you get a script execution error
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux/Git Bash:**
        ```bash
        source venv/bin/activate
        ```
    You should see `(venv)` at the beginning of your terminal prompt, indicating the environment is active.

4.  **Install Required Libraries:**
    With your virtual environment active, install all the project dependencies:
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn tensorflow
    ```

### How to Run the Application

1.  **Ensure your virtual environment is active** (you see `(venv)` in your terminal prompt).
2.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

3.  **Access the Application:**
    Your default web browser should automatically open a new tab displaying the Streamlit application. If not, open your browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

## 📊 Results

The application allows you to train and compare the performance of LSTM, GRU, and Hybrid LSTM-GRU models. The hybrid model is generally expected to perform better at extracting complex sequential information from voice data, leading to higher accuracy, precision, recall, and F1-score in detecting Parkinson's disease.

<img src="https://raw.githubusercontent.com/zuveriya-shaik/Parkinsons_Disease_Detection/refs/heads/main/Datasets/Screenshot%202025-06-19%20131610.png" alt="Streamlit App Screenshot - Model Comparison" width="700">

