import streamlit as st
from PIL import Image
import torch
import utils
import os
with open("styles/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# 1. Page Config
st.set_page_config(page_title="ResNet50 Baseline", page_icon="📊", layout="wide")

st.markdown("""
    <div>
            <h1 class = "hero-text">📊 Baseline Model: ResNet50</h1>
            <h3 class = "subheading">Standard Transfer Learning</h3>
    </div>
""",unsafe_allow_html = True)
st.warning("This model represents a standard approach. Notice how it might miss subtle cases compared to DenseNet.")

# 2. Sidebar Controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # A. Threshold
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.30, 0.05)
    st.sidebar.info("Note: Lower value of threshold increases the chance of pneumonia detection but also raises the false alarms.")
    
    st.divider()
    
    # B. Input Selection Method
    st.subheader("🖼️ Image Source")
    input_method = st.radio("Choose input method:", ["Upload Image", "Select Sample Case"])

    # C. The Gallery Logic
    selected_sample = None
    if input_method == "Select Sample Case":
        sample_folder = "test_samples"
        if os.path.exists(sample_folder):
            sample_files = [f for f in os.listdir(sample_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
            selected_sample = st.selectbox("Choose a test case:", sample_files)
        else:
            st.error("⚠️ 'test_samples' folder missing!")

# 3. Load Model (Cached)
# Ensure this filename matches your actual file
model = utils.load_resnet50("pneumonia_resnet_final.pth")

if model is None:
    st.warning("⚠️ Model file not found. Please upload 'pneumonia_resnet_final.pth'.")
    st.stop()

# 4. Handle Image Loading
image = None
image_source = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a Chest X-Ray", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_source = "Uploaded Image"

elif input_method == "Select Sample Case" and selected_sample:
    image_path = os.path.join("test_samples", selected_sample)
    image = Image.open(image_path).convert('RGB')
    image_source = f"Sample: {selected_sample}"

# 5. Run Analysis
if image:
    # A. Display Image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient X-Ray")
        st.image(image, caption=image_source, use_container_width=True)

    # B. Run Prediction
    with st.spinner("Analyzing lung patterns..."):
        transform = utils.get_transform()
        img_tensor = transform(image).unsqueeze(0)
        label, confidence = utils.predict(model, img_tensor, threshold)

    # C. Display Results
    with col2:
        st.subheader("AI Diagnosis")
        
        if label == "Pneumonia":
            st.error(f"## 🚨 **PNEUMONIA DETECTED**")
            st.metric("Confidence Score", f"{confidence:.1%}")
            st.markdown("The Baseline model suspects pneumonia.")
        else:
            st.success(f"## ✅ **NORMAL**")
            st.metric("Confidence Score", f"{(1-confidence):.1%}")
            st.markdown("Baseline model detects no issues.")
            

else:
    # Placeholder
    if input_method == "Upload Image":
        st.info("👆 Please upload an X-ray image to begin.")
        st.info("👈 OR select a sample case from the sidebar.")
    else:
        st.info("👈 Select a sample case from the sidebar.")