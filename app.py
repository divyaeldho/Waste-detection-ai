import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.set_page_config(page_title="Waste Detection App", layout="centered")

st.title("♻️ Waste Detection using YOLOv8")
st.write("Upload an image or capture from webcam to detect waste.")

@st.cache_resource
def load_model():
    return YOLO("models/Waste_detect_app.pt")

model = load_model()

# ---------- CONFIDENCE SLIDER ----------
conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# ---------- INPUT METHOD ----------
option = st.radio(
    "Choose Input Method:",
    ["Upload Image", "Capture from Webcam"],
    horizontal=True
)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", width=500)

if option == "Capture from Webcam":
    camera_image = st.camera_input("Capture image")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", width=500)

# ---------- DETECTION ----------
if image is not None and st.button("🔍 Detect Waste"):
    with st.spinner("Detecting waste..."):
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = model(img_bgr, conf=conf_threshold)

        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.success("Detection completed!")
    st.image(annotated_rgb, caption="Detection Result", width=500)

    # ---------- DETECTED OBJECT LIST ----------
    st.subheader("Detected Objects")
    if len(results[0].boxes) == 0:
        st.write("No waste detected.")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"• {model.names[cls_id]} — {conf:.2f}")

    # ---------- DOWNLOAD BUTTON ----------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="⬇️ Download Result Image",
            data=open(tmp.name, "rb"),
            file_name="waste_detection_result.jpg",
            mime="image/jpeg"
        )
