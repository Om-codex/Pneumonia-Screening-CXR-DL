import streamlit as st
from PIL import Image
import torch
import utils
import os
with open("styles/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# 1. Page Config
st.set_page_config(page_title="DenseNet121 Analysis", page_icon="🏆", layout="wide")

st.markdown("""
    <div>
            <h1 class = "hero-text">🏆 Champion Model: DenseNet121</h1>
            <h3 class = "subheading">High-Sensitivity Screening Mode</h3>
    </div>
""",unsafe_allow_html = True)

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
        # Get list of images from the 'test_samples' folder
        sample_folder = "test_samples"
        if os.path.exists(sample_folder):
            sample_files = [f for f in os.listdir(sample_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
            selected_sample = st.selectbox("Choose a test case:", sample_files)
        else:
            st.error("⚠️ 'test_samples' folder missing!")

# 3. Load Model
model = utils.load_densenet121("pneumonia_densenet_final.pth")

if model is None:
    st.warning("⚠️ Model file not found. Please check 'utils.py' or upload 'pneumonia_densenet_final.pth.")
    st.stop()

# 4. Handle Image Loading
image = None
image_source = None # To display the filename

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a Chest X-Ray", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_source = "Uploaded Image"

elif input_method == "Select Sample Case" and selected_sample:
    image_path = os.path.join("test_samples", selected_sample)
    image = Image.open(image_path).convert('RGB')
    image_source = f"Sample: {selected_sample}"

# 5. Run Analysis (Only if image exists)
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
            st.markdown("The model has detected signs of lung opacity.")
        else:
            st.success(f"## ✅ **NORMAL**")
            st.metric("Confidence Score", f"{(1-confidence):.1%}")
            st.markdown("No significant opacities detected.")

    st.divider()

    # D. Grad-CAM
    st.subheader("🧠 Explainable AI (Grad-CAM)")
    
    if st.toggle("Show Heatmap Analysis", value=True):
        with st.spinner("Generating attention map..."):
            target_layer = model.features.norm5 # The correct layer
            heatmap = utils.generate_gradcam(model, img_tensor, target_layer)
            overlay = utils.overlay_heatmap(heatmap, image)
            
            c1, c2 = st.columns(2)
            c1.image(heatmap, caption="Raw Attention Map", clamp=True, use_container_width=True)
            c2.image(overlay, caption="Overlay on X-Ray", clamp=True, use_container_width=True)
        
        st.warning("The **Red Zones** indicate the exact regions influencing the model's decision.")

else:
    # Placeholder
    if input_method == "Upload Image":
        st.info("👆 Please upload an X-ray image to begin.")
        st.info("👈 OR select a sample case from the sidebar.")
    else:
        st.info("👈 Select a sample case from the sidebar.")