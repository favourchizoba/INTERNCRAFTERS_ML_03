import streamlit as st
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load images from a folder and assign labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)).flatten()  # Resize and flatten the image
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load Cat and Dog images with correct labels
cat_images, cat_labels = load_images_from_folder('Cat', 0)  # 0 for Cat
dog_images, dog_labels = load_images_from_folder('Dog', 1)  # 1 for Dog

# Combine the data
X = np.concatenate((cat_images, dog_images), axis=0)
y = np.concatenate((cat_labels, dog_labels), axis=0)

# # Check the balance of the dataset
# def print_class_distribution(labels, label_names):
#     counts = np.bincount(labels)
#     for i, label_name in enumerate(label_names):
#         st.write(f"{label_name}: {counts[i]} samples")

# print_class_distribution(y, ['Cat', 'Dog'])

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Initialize and train the SVM model
model = SVC(kernel='linear', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model  
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app interface
# Title and header
st.markdown("<h1 style='color: #0C359E; text-align: center; font-size: 60px; font-family: Helvetica'>Cat and Dog Image Classification Using Support Vector Machine(SVM)</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.image('pngwing.com (33).png',width = 300) 

primaryColor="#FF4B4B"  
backgroundColor="#70E6D2"
secondaryBackgroundColor="#B4F2E8"
textColor="#31333F"
font="serif"

st.write("""
## User Guide: Cat and Dog Image Classification

Welcome to the Cat and Dog Image Classification app! This tool allows you to upload a photo or take a new one using your device's camera to identify whether the image contains a cat or a dog. Follow the steps below to use the app effectively:

### Step 1: Choose an Option
- **Upload a Photo:** If you have an image stored on your device, select this option. Supported formats are `.jpg`, `.jpeg`, and `.png`.
- **Take a Photo:** If you want to capture a new image using your device's camera, select this option.

### Step 2: Upload or Capture an Image
- **Upload a Photo:** 
  1. Click the "Choose an image..." button.
  2. Browse your device's files to find the image you want to upload.
  3. Once selected, the image will automatically be processed by the model.
  
- **Take a Photo:**
  1. Click the "Capture a photo" button.
  2. Allow the app to access your camera.
  3. Capture the image by following the on-screen instructions.
  4. The captured image will automatically be processed by the model.

### Step 3: View the Prediction
- After uploading or capturing an image, the app will display the image along with the predicted label (either "Cat" or "Dog").
- The prediction will appear below the image as text.

### Notes:
- Ensure the image is clear and well-lit for the best results.
- The model may not perform as expected on images with multiple animals or other objects.

Thank you for using the Cat and Dog Image Classification app!
""")

# st.write(f"Model accuracy: {accuracy * 100:.2f}%")

# Function to process and classify images
def process_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_flattened = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_flattened)
    return "Dog" if prediction == 1 else "Cat"

# Option to upload an image or use the camera
option = st.selectbox("Choose an option:", ["Upload a Photo", "Take a Photo"])

if option == "Upload a Photo":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        label = process_image(image)
        st.image(image, channels="BGR")
        st.write(f"This is a {label}")

elif option == "Take a Photo":
    camera_image = st.camera_input("Capture a photo")
    if camera_image is not None:
        # Convert the camera image to OpenCV format
        image = np.array(camera_image)
        label = process_image(image)
        st.image(image, channels="BGR")
        st.write(f"This is a {label}")  



