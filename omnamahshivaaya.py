import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('traffic_classifier.h5')

# Dictionary to label all traffic signs classes
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing for vehicles over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Vehicles > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End of speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing for vehicles > 3.5 tons'
}

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='#CDCDCD')

label = tk.Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = tk.Label(top)

def classify(file_path):
    """Classify the uploaded image."""
    global label_packed
    try:
        image = Image.open(file_path).convert('RGB')  # Ensure 3-channel RGB format
        image = image.resize((30, 30))  # Resize to model input size
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.array(image) / 255.0  # Normalize
        probabilities = model.predict(image)
        pred = np.argmax(probabilities, axis=-1)[0]
        confidence = probabilities[0, pred]

        if confidence < 0.6:  # Confidence threshold
            label.configure(foreground='red', text="Prediction Uncertain!")
        else:
            sign = classes.get(pred + 1, "Unknown Sign")
            label.configure(foreground='#011638', text=f"{sign} ({confidence*100:.2f}% confident)")
    except Exception as e:
        print(f"Error during classification: {e}")
        label.configure(foreground='red', text="Error in classification!")


def show_classify_button(file_path):
    """Show the 'Classify' button after an image is uploaded."""
    classify_b = tk.Button(
        top, text="Classify Image",
        command=lambda: classify(file_path),
        padx=10, pady=5
    )
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    """Upload an image and display it."""
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            uploaded = Image.open(file_path).convert('RGB')  # Convert to RGB
            uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=im)
            sign_image.image = im
            label.configure(text='')
            show_classify_button(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        label.configure(foreground='red', text="Error loading image!")

# GUI Layout
upload = tk.Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=tk.BOTTOM, pady=50)

sign_image.pack(side=tk.BOTTOM, expand=True)
label.pack(side=tk.BOTTOM, expand=True)

heading = tk.Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()

