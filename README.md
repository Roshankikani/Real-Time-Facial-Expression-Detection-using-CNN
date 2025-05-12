# Real-Time Facial Expression Recognition using CNN

This project implements a real-time facial expression recognition system using a Convolutional Neural Network (CNN) trained on grayscale facial images. The system captures facial expressions via webcam, detects faces using Haar cascades, and classifies emotions into seven distinct categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## 📌 Overview

Facial expression is a key indicator of human emotions. This project focuses on identifying these emotions using deep learning, specifically leveraging a CNN architecture trained on 48x48 grayscale images.

## 🔧 Model Architecture

The CNN model was constructed using Keras and TensorFlow backend. It includes the following components:

- **Input Shape**: (48, 48, 1)
- **Convolutional Layers**: Four Conv2D layers with filters ranging from 32 to 128.
- **Batch Normalization**: Applied after each convolution to stabilize learning.
- **Activation**: ReLU is used for all hidden layers; Softmax in the output layer.
- **Dropout**: Applied to prevent overfitting (rates of 0.25 and 0.5).
- **MaxPooling**: Reduces spatial dimensions progressively.
- **Dense Layers**: One fully connected layer with 250 units and one output layer with 7 units.

This model is encapsulated in a `Sequential` structure and uses `categorical_crossentropy` as the loss function.

> 🧠 **Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 📈 Accuracy

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~65% (indicative, depends on dataset preprocessing and balance)

## 🎥 Real-Time Detection (`webcam_test.py`)

This script initiates webcam access and performs real-time facial expression recognition. Here's how it works:

1. **Model Loading**:
   - Loads model architecture from `Facial Expression Recognition.json`
   - Loads trained weights from `fer.h5`

2. **Face Detection**:
   - Uses `haarcascade_frontalface_default.xml` to detect face regions.

3. **Preprocessing**:
   - Converts image to grayscale.
   - Crops and resizes the detected face to 48x48 pixels.
   - Normalizes pixel values.

4. **Prediction**:
   - Passes the preprocessed face image into the CNN model.
   - Retrieves the predicted emotion and displays it on the video feed.

5. **Visualization**:
   - Draws bounding boxes and emotion labels over the detected faces.

> Press **'q'** to quit the webcam interface.

## 📁 Files

- `webcam_test.py`  
  Real-time detection script using webcam and CNN model.

- `Real-Time Facial Expression Detection.ipynb`  
  Jupyter Notebook for training, validating, and evaluating the CNN model.

- `Facial Expression Recognition.json`  
  Serialized CNN architecture in JSON format.

- `fer.h5`  
  Pre-trained model weights for facial emotion classification.

- `haarcascade_frontalface_default.xml`  
  Haar cascade classifier for frontal face detection using OpenCV.


## 🧠 Core Functions (for reference)

| Function | Description |
|---------|-------------|
| `model_from_json()` | Loads CNN architecture from serialized JSON. |
| `CascadeClassifier()` | Initializes Haar cascade for face detection. |
| `detectMultiScale()` | Detects objects (faces) in grayscale frames. |
| `model.predict()` | Performs emotion classification on face images. |


## 🚀 How to Run

```bash
pip install -r requirements.txt
python webcam_test.py
