# ==============================================================================
# app.py
#
# Streamlit demo for the surface defect detection model.
# Click a sample image to classify it, or upload your own.
# ==============================================================================

import io
import base64
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

# Third-party component for clickable images
try:
    from st_clickable_images import clickable_images
except Exception:
    clickable_images = None

# --------------------------- Model loading ------------------------------------
@st.cache_resource
def load_model():
    model_path = "saved_models/surface_defect_detector_best.pth"
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# --------------------------- Inference ----------------------------------------
def predict(image_bytes: bytes) -> str:
    model = load_model()
    class_names = [
        "Crazing",
        "Inclusion",
        "Patches",
        "Pitted_Surface",
        "Rolled-in_Scale",
        "Scratches",
    ]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        _, pred_idx = torch.max(model(x), 1)
    return class_names[pred_idx.item()]

# --------------------------- App UI -------------------------------------------
st.set_page_config(page_title="AI Defect Detector", layout="wide")
st.title("AI Surface Defect Detector")
st.write(
    "This app demonstrates a model trained to identify 6 types of surface defects. "
    "First, try a sample image to see the model's accuracy. Then, upload your own!"
)
st.markdown("---")

# --------------------------- Sample images ------------------------------------
st.subheader("1. Try a Sample Image from the Dataset")

BASE_DIR = Path(__file__).resolve().parent
DEMO_IMAGES = BASE_DIR / "demo_images"

if not DEMO_IMAGES.exists():
    st.error("The 'demo_images' folder was not found. Please create it and add some sample images.")
else:
    sample_paths = sorted(
        [p for p in DEMO_IMAGES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )

    if not sample_paths:
        st.warning("No sample images found in 'demo_images'.")
    else:
        if clickable_images is None:
            st.error(
                "Install the clickable images component first:\n\n"
                "  pip install st-clickable-images\n\n"
                "and add `st-clickable-images>=0.0.3` to requirements.txt."
            )
        else:
            if "selected_idx" not in st.session_state:
                st.session_state.selected_idx = None  # nothing selected at startup

            # Load images & captions
            images = [Image.open(p).convert("RGB") for p in sample_paths]
            captions = [p.name for p in sample_paths]

            # Helper: convert PIL -> data URI for clickable_images
            def pil_to_data_uri(img):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/png;base64,{b64}"

            data_uris = [pil_to_data_uri(img) for img in images]

            # Render gallery (returns -1 until user clicks)
            clicked_idx = clickable_images(
                data_uris,
                titles=captions,
                div_style={
                    "display": "grid",
                    "grid-template-columns": "repeat(auto-fit, minmax(160px, 1fr))",
                    "gap": "10px",
                },
                img_style={"width": "100%", "border-radius": "8px"},
                key="samples_gallery",
            )

            if clicked_idx > -1:
                st.session_state.selected_idx = clicked_idx

            chosen_idx = st.session_state.selected_idx
            if chosen_idx is not None:
                chosen_path = sample_paths[chosen_idx]
                with open(chosen_path, "rb") as f:
                    img_bytes = f.read()
                with st.spinner("Classifying..."):
                    pred = predict(img_bytes)

                # Show prediction directly under the clicked image
                cols = st.columns(len(sample_paths))
                with cols[chosen_idx]:
                    st.success(f"**Predicted:** {pred}")

st.markdown("---")

# --------------------------- Uploader -----------------------------------------
st.subheader("2. Or, Upload Your Own Image")
st.write("For best results, upload a clear, top-down image of a **metal surface**.")
uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    img_bytes = uploaded.getvalue()
    st.image(img_bytes, caption="Uploaded Image.", use_container_width=True)
    with st.spinner("Classifying..."):
        model = load_model()
        class_names = [
            "Crazing",
            "Inclusion",
            "Patches",
            "Pitted_Surface",
            "Rolled-in_Scale",
            "Scratches",
        ]
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

    st.success(f"**Predicted Defect:** {class_names[pred_idx]}")

    # 👇 Add this line to show full probability distribution
    # st.write({cls: float(probs[i]) for i, cls in enumerate(class_names)})

    

# --------------------------- Notes --------------------------------------------
st.markdown("---")
st.subheader("How to Interpret the Prediction")
st.write(
    """
    This model acts like a **specialist** trained on the specific NEU Surface Defect dataset. Its performance depends on how similar the uploaded image is to the data it learned from.

    - **When the prediction is correct:** This happens when the features in the uploaded image (like lighting, texture, and angle) closely match the patterns the model learned during its training. This is especially true for the sample images, where the model demonstrates its high accuracy (>95%) on familiar data.

    - **When the prediction is incorrect:** This usually occurs if you upload a random photo from the internet that is "out-of-distribution." The lighting or texture might be too different for the model's specialized knowledge to apply. This is a classic real-world AI challenge, and the solution is to train the model on a larger, more diverse dataset.
   
    """
)
