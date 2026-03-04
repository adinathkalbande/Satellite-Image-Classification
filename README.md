# 🛰 Satellite Image Classifier

A Deep Learning based web application that classifies satellite images into different land cover categories using a Convolutional Neural Network (CNN). The application is built with **TensorFlow/Keras** for the model and **Streamlit** for the interactive web interface.

---

## 📌 Project Overview

Satellite imagery plays an important role in land monitoring, agriculture, urban planning, and environmental analysis. This project uses a CNN model trained on the **EuroSAT dataset** to automatically classify satellite images into different land cover classes.

The system allows users to upload a satellite image through a web interface and instantly get the predicted land cover type with confidence score.

---

## 🎯 Features

* 🛰 Satellite image classification using Deep Learning
* 🧠 CNN model trained on the EuroSAT dataset
* 📤 Upload satellite images through a web interface
* 🎨 Interactive UI built with Streamlit
* 📊 Confidence score displayed for predictions
* ⚡ Fast prediction with cached model loading

---

## 🗂 Dataset

This project uses the **EuroSAT RGB dataset**, which contains satellite images categorized into 10 land cover classes.

Classes in the dataset:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

Dataset Link:
https://github.com/phelber/eurosat

---

## 🧠 Model Architecture

The classification model is built using a **Convolutional Neural Network (CNN)**.

Typical architecture:

* Convolution Layer
* ReLU Activation
* Max Pooling
* Convolution Layer
* Max Pooling
* Flatten Layer
* Dense Layer
* Output Layer (Softmax)

Input image size used for training:

```
64 × 64 × 3
```

---

## 🖥 Application Interface

The application interface allows users to:

1. Upload a satellite image
2. View the image preview
3. Get predicted land cover class
4. See prediction confidence score

The UI is developed using **Streamlit** with custom styling.

---

## 📂 Project Structure

```
Satellite-Image-Classifier
│
├── app.py                # Streamlit application
├── eurosat_model.h5      # Trained CNN model
├── dataset               # Training dataset (optional)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## 📦 Dependencies

Main libraries used in this project:

* TensorFlow
* Keras
* NumPy
* Streamlit
* Pillow

Install them using:

```
pip install tensorflow streamlit numpy pillow
```

---

## 🚀 Future Improvements

* Deploy the application on **Streamlit Cloud**
* Add **Top-3 predictions visualization**
* Improve model accuracy using **Transfer Learning**
* Add **Grad-CAM visualization for explainable AI**

---

## 👨‍💻 Author

Developed by **Adinath Kalbande**

---

## 📜 License

This project is for educational and research purposes.
