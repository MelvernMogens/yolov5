import os
import requests
import torch
import streamlit as st
import numpy as np
from PIL import Image
import cv2

# URL to the YOLOv5 model file on GitHub
MODEL_URL = "https://github.com/MelvernMogens/yolov5/raw/main/best.pt"
MODEL_PATH = "best.pt"

# Function to download the YOLOv5 model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLOv5 model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully!")

# Download the model
download_model()

# Load YOLOv5 model with caching to avoid reloading
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        st.stop()

model = load_model()

# Function to run YOLOv5 on an image
def detect_image(image):
    """Run YOLOv5 model on the provided image."""
    results = model(image)
    return results

# Streamlit UI
st.title("YOLOv5 Object Detection")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Image Detection", "Webcam Detection"])

if page == "Image Detection":
    st.header("Upload an Image for Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Detecting...")
        
        # Convert PIL Image to numpy
        image_np = np.array(image)
        
        # Run YOLOv5 on the image
        results = detect_image(image_np)
        
        # Display results
        st.image(np.squeeze(results.render()), caption="Detected Image", use_container_width=True)

elif page == "Webcam Detection":
    st.header("Webcam Detection with YOLOv5")
    webcam_index = st.number_input("Enter Webcam Index", value=0, step=1)
    start_detection = st.button("Start Detection")

    if start_detection:
        cap = cv2.VideoCapture(webcam_index)
        st_frame = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to access webcam")
                break

            # YOLOv5 detection
            results = detect_image(frame)

            # Render results on the frame
            detected_frame = np.squeeze(results.render())

            # Display the frame in Streamlit
            st_frame.image(detected_frame, channels="BGR", use_container_width=True)

        cap.release()
