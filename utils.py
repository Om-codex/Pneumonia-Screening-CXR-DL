import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import streamlit as st

# --- 1. CONFIGURATION ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- 2. MODEL LOADERS ---
@st.cache_resource
def load_resnet50(filepath):
    device = torch.device("cpu")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
      nn.Linear(num_ftrs,512),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      nn.Linear(512, 2)
    )
    
    try:
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ResNet: {e}")
        return None

@st.cache_resource
def load_densenet121(filepath):
    device = torch.device("cpu")
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    
    # Rebuild the head exactly as trained
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    
    try:
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading DenseNet: {e}")
        return None

# ------3. GradCAM-----
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # FIX: Use only a forward hook. We catch gradients via a Tensor Hook.
        self.handle = target_layer.register_forward_hook(self.save_activation_and_hook)

    def save_activation_and_hook(self, module, input, output):
        # 1. Save the activation (feature map)
        # Use .detach().clone() to avoid in-place modification errors
        self.activations = output.detach().clone()
        
        # 2. Hook the TENSOR directly to catch gradients
        # This avoids the "BackwardHookFunctionBackward" error
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        # We must enable gradients for this pass, even if the App is in "no_grad" mode
        with torch.enable_grad():
            # 1. Forward Pass
            output = self.model(x)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1)
            
            # 2. Backward Pass (Calculate Importance)
            self.model.zero_grad()
            class_score = output[0, class_idx]
            class_score.backward()
            
            # 3. Generate Heatmap
            if self.gradients is None or self.activations is None:
                return np.zeros((224, 224)) # Safety return

            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            
            # Multiply features by importance
            activations = self.activations[0]
            for i in range(activations.shape[0]):
                activations[i, :, :] *= pooled_gradients[i]
                
            # Create heatmap
            heatmap = torch.mean(activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0) # ReLU
            
            # Normalize
            if np.max(heatmap) != 0:
                heatmap /= np.max(heatmap)
                
            return heatmap

    # Cleanup hook (Optional but good practice)
    def remove_hooks(self):
        self.handle.remove()

def generate_gradcam(model, image_tensor, target_layer):
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(image_tensor, class_idx=1) 
    grad_cam.remove_hooks() # Clean up
    return heatmap

def overlay_heatmap(heatmap, original_image):
    # Convert PIL to numpy array (RGB)
    img_np = np.array(original_image)
    
    # Resize heatmap to match image size (Using CUBIC for smoothness)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Colorize
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# --- 4. PREDICTION HELPER ---
def predict(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        p_pneumonia = probs[0][1].item()
        
        prediction = "Pneumonia" if p_pneumonia > threshold else "Normal"
        return prediction, p_pneumonia
