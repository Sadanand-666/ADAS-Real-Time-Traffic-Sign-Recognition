# Import Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
from PIL import Image 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout 

# Constants 
IMG_HEIGHT, IMG_WIDTH = 30, 30 
NUM_CLASSES = 43 
EPOCHS = 15 
BATCH_SIZE = 32 

# Load and Preprocess Data 
dataset_path = "/content/drive/MyDrive/gtsrb-german-traffic-sign/Project/archivedproject/GTSRB---German-Traffic-Sign-Recognition-main/gtsrb-german-traffic-sign/Train"  # Update this path to your dataset folder 

data = [] 
labels = [] 

for i in range(NUM_CLASSES): 
    path = os.path.join(dataset_path, str(i)) 
    images = os.listdir(path) 
    for img_name in images: 
        try: 
            img_path = os.path.join(path, img_name) 
            image = Image.open(img_path) 
            image = image.resize((IMG_HEIGHT, IMG_WIDTH)) 
            data.append(np.array(image)) 
            labels.append(i) 
        except Exception as e: 
            print(f"Error loading image {img_name}: {e}") 

# Convert lists to numpy arrays 
data = np.array(data) 
labels = np.array(labels) 
print(f"Data shape: {data.shape}, Labels shape: {labels.shape}") 

# Split into Training and Testing datasets 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42) 
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Labels: {y_train.shape}, {y_test.shape}") 

# Normalize the data 
X_train = X_train / 255.0 
X_test = X_test / 255.0 

# One-hot encoding for labels 
y_train = to_categorical(y_train, NUM_CLASSES) 
y_test = to_categorical(y_test, NUM_CLASSES) 

# Define the Model 
model = Sequential([ 
    Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), 
    Conv2D(32, (5, 5), activation='relu'), 
    MaxPool2D(pool_size=(2, 2)), 
    Dropout(0.25), 
    Conv2D(64, (3, 3), activation='relu'), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPool2D(pool_size=(2, 2)), 
    Dropout(0.25), 
    Flatten(), 
    Dense(256, activation='relu'), 
    Dropout(0.5), 
    Dense(NUM_CLASSES, activation='softmax'), 
]) 

# Compile the Model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Train the Model 
history = model.fit( 
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS 
) 

# Save the Model 
model.save("traffic_classifier.h5") 
print("Model saved as traffic_classifier.h5") 

# Load and Recompile the Model 
from keras.models import load_model 
model = load_model('traffic_classifier.h5') 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

# Save the model in .keras format 
from keras.saving import save_model 
save_model(model, 'my_model.keras') 

# Plot Training Accuracy and Loss 
plt.figure(0) 
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show() 

plt.figure(1) 
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.title('Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 
plt.show()
