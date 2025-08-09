import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
from torchvision import models
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Breast Cancer Detection Assistant",
    page_icon="🏥",
    layout="wide",
)

# --- Model Definition (Must be identical to the training script) ---
class BreastCancerModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(BreastCancerModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# --- Helper Functions ---
@st.cache_resource
def load_model_and_encoder(model_path='breast_cancer_model_v3.pth'):
    """Loads the trained 3-class model and label encoder from a checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        label_encoder = checkpoint['label_encoder']
        num_classes = len(label_encoder.classes_)
        
        model = BreastCancerModel(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, label_encoder
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'. Please train the new 3-class model first.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

@st.cache_data
def load_segmentation_data(csv_path='dataset/Radiology_hand_drawn_segmentations.csv'):
    """Loads and parses the segmentation data from the CSV file."""
    try:
        df = pd.read_csv(csv_path, header=0)
        df = df.rename(columns={df.columns[0]: 'filename'})

        segmentations = {}
        for _, row in df.iterrows():
            filename = row['filename']
            if filename not in segmentations:
                segmentations[filename] = []
            
            region_data = json.loads(row['region_shape_attributes'])
            segmentations[filename].append(region_data)
        return segmentations
    except FileNotFoundError:
        st.warning(f"Segmentation file not found at '{csv_path}'. The segmentation overlay feature will be disabled.")
        return None
    except Exception as e:
        st.warning(f"Could not load or parse segmentation data: {e}")
        return None

def preprocess_image(image):
    """Prepares the uploaded image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def draw_segmentations(original_image, segmentation_regions):
    """Draws segmentation masks on the image."""
    img_with_drawing = original_image.copy()
    draw = ImageDraw.Draw(img_with_drawing)
    
    for region in segmentation_regions:
        shape_name = region.get('name')
        
        if shape_name == 'polygon':
            points = list(zip(region['all_points_x'], region['all_points_y']))
            draw.polygon(points, outline="yellow", width=5)
        elif shape_name == 'ellipse':
            cx, cy, rx, ry = region['cx'], region['cy'], region['rx'], region['ry']
            bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
            draw.ellipse(bbox, outline="yellow", width=5)
        elif shape_name == 'circle':
            cx, cy, r = region['cx'], region['cy'], region['r']
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, outline="yellow", width=5)
            
    return img_with_drawing

# --- Main App ---
def main():
    st.title("🏥 Breast Cancer Detection Assistant")
    st.warning("⚠️ **Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified doctor.")

    # Load Model and Data
    model, label_encoder = load_model_and_encoder()
    segmentation_data = load_segmentation_data()
    
    if model is None:
        return

    # Sidebar
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose a mammogram image...", type=["jpg", "jpeg", "png"])
        st.header("Model Information")
        st.info(f"**Model:** EfficientNet-B0 (Fine-tuned)\n\n**Classes:** {', '.join(label_encoder.classes_)}")

    # Main Panel
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            # --- FIX: Replaced use_column_width with use_container_width ---
            st.image(image, caption='Uploaded Mammogram', use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing the image...'):
                image_tensor = preprocess_image(image)
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)
                    predicted_class = label_encoder.classes_[predicted_idx]

            st.subheader("Analysis Results")
            if predicted_class == 'Malignant':
                st.error(f"**Prediction:** Malignant")
            elif predicted_class == 'Benign':
                st.warning(f"**Prediction:** Benign")
            else: # Normal
                st.success(f"**Prediction:** Normal")
            
            st.metric(label="Confidence", value=f"{confidence.item():.2%}")
            
            st.subheader("📊 Class Probabilities")
            for i, class_name in enumerate(label_encoder.classes_):
                prob = probabilities[i].item()
                st.write(f"**{class_name}:**")
                st.progress(prob)

        # --- Segmentation Overlay Section ---
        if segmentation_data and uploaded_file.name in segmentation_data:
            st.subheader("🩺 Expert Annotation Overlay")
            regions = segmentation_data[uploaded_file.name]
            image_with_segmentation = draw_segmentations(image, regions)
            # --- FIX: Replaced use_column_width with use_container_width ---
            st.image(image_with_segmentation, caption="Image with expert-drawn lesion boundaries", use_container_width=True)
        else:
            st.info("No expert segmentation data was found for this specific image file.")
            
    else:
        st.info("Please upload an image to begin analysis.")

if __name__ == "__main__":
    main()
