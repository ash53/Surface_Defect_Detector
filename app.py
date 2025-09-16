# ==============================================================================
# app.py
#
# A Streamlit web application to demo the surface defect detection model.
# Allows users to upload an image or test with provided samples.
#
# ==============================================================================

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os

# --- 1. Model Loading and Caching ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """
    Loads and returns the pre-trained PyTorch model.
    """
    model_path = 'saved_models/surface_defect_detector_best.pth'
    # Initialize the model architecture (ResNet18 with 6 output classes)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6) # We have 6 defect classes
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval() # Set the model to evaluation mode
    return model

# --- 2. Prediction Function ---
def predict(image_bytes):
    """
    Takes an image in bytes, preprocesses it, and returns the predicted class name.
    """
    model = load_model()
    class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled-in_Scale', 'Scratches']
    # Define the same image transformations as the validation set
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Preprocess the image and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    return class_names[predicted_idx.item()]

# --- 3. Streamlit App ---

# Set up the main page layout and title
st.set_page_config(page_title="AI Defect Detector", layout="wide")
st.title("AI Surface Defect Detector")
st.write(
    "This app demonstrates a model trained to identify 6 types of surface defects. "
    "First, try a sample image to see the model's accuracy. Then, upload your own!"
)
st.markdown("---")


# --- 4. Sample Image Section ---
st.subheader("1. Try a Sample Image from the Dataset")

# Find sample images in the 'demo_images' folder
demo_images_path = "demo_images"
try:
    sample_images = [f for f in os.listdir(demo_images_path) if f.endswith(('.png', '.jpg', '.bmp'))]
    
    if sample_images:
        # Create columns for each sample image
        cols = st.columns(len(sample_images))
        for i, image_name in enumerate(sample_images):
            with cols[i]:
                image_path = os.path.join(demo_images_path, image_name)
                st.image(image_path, caption=image_name, use_column_width=True)
                
                # Each button has a unique key to differentiate it
                if st.button(f"Classify {image_name}", key=image_name):
                    with st.spinner("Classifying..."):
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()
                        prediction = predict(image_bytes)
                        st.success(f"**Predicted:** {prediction}")
    else:
        st.warning("No sample images found. Please add some images to the 'demo_images' folder.")

except FileNotFoundError:
    st.error("The 'demo_images' folder was not found. Please create it and add some sample images.")

st.markdown("\n---")


# --- 5. File Uploader Section ---
st.subheader("2. Or, Upload Your Own Image")
st.write("For best results, upload a clear, top-down image of a **metal surface**.")

# The file uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "bmp"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption='Uploaded Image.', use_column_width=True)

    # Show a spinner while classifying
    with st.spinner("Classifying..."):
        prediction = predict(image_bytes)
        st.success(f"**Predicted Defect:** {prediction}")


# --- 6. Explanatory Section for Recruiters ---
st.markdown("---")
st.subheader("How to Interpret the Prediction")
st.write(
    """
    This model acts like a **specialist** trained on the specific NEU Surface Defect dataset. Its performance depends on how similar the uploaded image is to the data it learned from.

    - **When the prediction is correct:** This happens when the features in the uploaded image (like lighting, texture, and angle) closely match the patterns the model learned during its training. This is especially true for the sample images, where the model demonstrates its high accuracy (>95%) on familiar data.

    - **When the prediction is incorrect:** This usually occurs if you upload a random photo from the internet that is "out-of-distribution." The lighting or texture might be too different for the model's specialized knowledge to apply. This is a classic real-world AI challenge, and the solution is to train the model on a larger, more diverse dataset.
    """
)

