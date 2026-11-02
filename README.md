# 🥬 SmartPantry – Food Waste Risk Prediction System

**SmartPantry** is a Machine Learning-based application that estimates food waste risk using food type, storage method, storage duration, remaining quantity, temperature and humidity.

It classifies the estimated waste risk into three categories: **Low, Medium and High**, and provides simple recommendations through an interactive web interface.

## ✨ Features

- Food waste risk prediction using Machine Learning.
- Exploratory Data Analysis and data visualisation.
- Decision Tree classification model.
- Prediction confidence and recommendations.
- Interactive Gradio web application.
- Deployment using Hugging Face Spaces.

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Gradio
- Google Colab
- Hugging Face Spaces

## 📊 Dataset

The project uses a synthetically generated dataset of approximately 1,200 records.

**Input features:** Food Type, Storage Type, Days Stored, Quantity Remaining, Temperature and Humidity.

**Target variable:** Waste Risk (Low, Medium, High).

## ⚙️ Machine Learning Workflow

1. Dataset loading and cleaning.
2. Exploratory Data Analysis (EDA).
3. Feature selection and preprocessing.
4. Train-test splitting (80:20).
5. Decision Tree model training.
6. Model evaluation using accuracy, precision, recall, F1-score and confusion matrix.
7. Model saving using Joblib.
8. Gradio application development and deployment.

## 🌐 Live Application

🚀 [Try SmartPantry on Hugging Face](https://huggingface.co/spaces/aaryakelkar11/SmartPantry)

## 📁 Project Structure

```text
SmartPantry/
├── SmartPantry.ipynb
├── SmartPantry_Dataset.csv
├── SmartPantry_Model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## 🔮 Future Scope

- Integration of real-world food spoilage datasets.
- Improved prediction using additional ML algorithms.
- IoT-based temperature and humidity monitoring.
- Expiry-date tracking and inventory notifications.
- Improved model confidence calibration.

## ⚠️ Limitations

Predictions estimate food waste risk and do not determine actual food freshness, expiry or safety.

---

<p align="center">
  <b>SmartPantry – Making Food Inventory Management Smarter 🌱</b>
</p>
