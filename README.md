


# ADAS-Real-Time-Traffic-Sign-Recognition
"ADAS: Real-Time Traffic Sign Recognition" is an advanced system that detects and classifies traffic signs using deep learning and computer vision. It enhances driver awareness by providing real-time alerts, improving road safety through accurate recognition of speed limits, warnings, and other critical signs.



## Overview
**ADAS: Real-Time Traffic Sign Recognition** is a deep learning–based project designed to detect and classify traffic signs in real time. It enhances driver assistance systems by recognizing various road signs—such as speed limits, warnings, and prohibitions—and providing immediate alerts to improve road safety and situational awareness.

This project was developed as part of a college group project, focusing on combining computer vision and machine learning for intelligent transportation systems.

---

## Features
- Real-time detection and classification of traffic signs  
- Deep learning model trained using **TensorFlow** and **Keras**  
- Integrated **OpenCV** for live video and image frame processing  
- Supports real-time camera input or pre-recorded videos  
- Optimized model performance with data augmentation and preprocessing  
- Model compression applied for faster inference and embedded deployment  

---

## Technologies Used
- **Programming Language:** Python  
- **Frameworks & Libraries:** TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib  
- **Model Architecture:** CNN (Convolutional Neural Network)  
- **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  

---

## System Architecture
1. **Data Collection & Preprocessing**
   - Loaded and normalized image data.
   - Applied augmentation (rotation, zoom, brightness) to improve model generalization.
2. **Model Training**
   - CNN-based deep learning model trained on labeled traffic sign images.
3. **Model Evaluation**
   - Measured accuracy, precision, and recall on test data.
4. **Real-Time Detection**
   - Integrated OpenCV to capture frames from a webcam.
   - Detected and classified signs on live video streams.
5. **Alert System**
   - Displayed recognized traffic sign and label overlay on screen in real time.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ADAS-Real-Time-Traffic-Sign-Recognition.git
   cd ADAS-Real-Time-Traffic-Sign-Recognition

