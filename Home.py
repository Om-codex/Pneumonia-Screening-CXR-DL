import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import utils

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)
with open("styles/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 1. Page Configuration (Must be the first line)
st.set_page_config(
    page_title="PulmoScan AI",
    page_icon="🫁",
    layout="wide"
)
# 2. Hero Section

st.markdown("""
    <div>
            <h1 class = "hero-text">🫁 PulmoScan AI: Pneumonia Screening</h1>
            <h3 class = "subheading">A Comparative Study of Deep Learning Architectures for Medical Imaging</h3>
    </div>""",unsafe_allow_html = True
    )
st.markdown("""
    **Author:** Om Santosh Mishra  
    **Objective:** To build a high-sensitivity screening tool that minimizes missed diagnoses (False Negatives).

""")
# ----sidebar----
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "ResNet50 Baseline", "DenseNet121 Champion"],
        icons=["house", "bar-chart", "trophy"],
        menu_icon="cast",
        default_index=0
    )
if selected == "ResNet50 Baseline":
    st.switch_page("pages/ResNet50_Baseline.py")
elif selected == "DenseNet121 Champion":
    st.switch_page("pages/DenseNet121_Champion.py")




st.divider()

# 3. The "Why" - Problem Statement
col1, col2 = st.columns(2)

with col1:
    st.info("🎯 **The Goal**")
    st.markdown("""
    Pneumonia is a life-threatening condition where early detection is critical. 
    In medical AI, **Recall (Sensitivity)** is the most important metric because **missing a sick patient (False Negative) is dangerous**, whereas flagging a healthy one (False Positive) is just an inconvenience.
    """)

with col2:
    st.warning("⚠️ **The Challenge**")
    st.markdown("""
    We fine-tuned two powerful architectures, **ResNet50** and **DenseNet121**, under identical conditions.
    While both achieved high accuracy, we analyzed which model was "safer" for deployment in a real-world clinical setting.
    """)

# 4. The "Proof" - Key Results
st.header("📊 Model Benchmark Results")
st.markdown("We tested both models on a held-out test set of **624 X-Ray images**.")

# Metric Cards
m1, m2, m3 = st.columns(3)
m1.metric(label="✅ Champion Model", value="DenseNet121", delta="Selected")
m2.metric(label="🛡️ Safety Score (Recall)", value="99.7%", delta="+2.7% vs Baseline")
m3.metric(label="📉 Missed Cases", value="1 / 390", delta="-10 vs Baseline")

# Comparison Table
st.subheader("Head-to-Head Comparison")
# Updated Comparison Table in Home.py
comparison_data = {
    "Metric": ["Architecture", "Recall (Sensitivity)", "False Alarms", "Verdict"],
    "Model A": ["ResNet50", "97.0%", "High", "❌ Unreliable"],
    "Model B": ["DenseNet121 (Champion)", "100%", "Low", "✅ Superior Performance"]
}
df = pd.DataFrame(comparison_data)
st.table(df)

st.divider()

# 5. Application Guide
st.header("🛈 How to Use This App")
st.subheader("Use the **Sidebar Menu** to test the models:")
st.markdown("""

1.  **📊 ResNet50 Analysis:** Test the fine-tuned ResNet model. While accurate, notice how it occasionally misses subtle pneumonia features.
2.  **🏆 DenseNet121 Analysis:** Test the champion model. Includes **Grad-CAM** explanation to visualize the lung opacities triggering the diagnosis.
""")

# 5. Direct Navigation Buttons
st.header("✨ Try the Models")
st.subheader("Select a model to start the diagnostic simulation:")

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    st.markdown("### Option 1: The Baseline")
    st.markdown("See where standard transfer learning struggles.")

    if st.button("📊 Go to ResNet50 Analysis", use_container_width=True):
        st.switch_page("pages/ResNet50_Baseline.py")

with btn_col2:
    st.markdown("### Option 2: The Champion")
    st.markdown("Experience the high-sensitivity model with Grad-CAM.")

    if st.button("🏆 Go to DenseNet121 Analysis", use_container_width=True):
        st.switch_page("pages/DenseNet121_Champion.py")

# 6. Tech Stack Footer
st.markdown("---")
st.caption("🛠️ **Tech Stack:** PyTorch | Streamlit | OpenCV (Grad-CAM) | Albumentations")