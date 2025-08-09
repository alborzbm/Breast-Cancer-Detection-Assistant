import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import argparse
from torchvision import models
import warnings
warnings.filterwarnings('ignore')

# --- Model Definition (MUST be identical to train_model.py and app.py) ---
class BreastCancerModel(nn.Module):
    def __init__(self, num_classes):
        super(BreastCancerModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Fine-tuning: Unfreeze the last few layers
        for param in self.base_model.features[-3:].parameters():
            param.requires_grad = True
            
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# --- Helper Functions ---
def load_pytorch_model(model_path):
    """Loads a trained PyTorch model and its label encoder from a checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        label_encoder = checkpoint['label_encoder']
        num_classes = len(label_encoder.classes_)
        
        model = BreastCancerModel(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully from '{model_path}'")
        return model, label_encoder
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please run train_model.py first.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None

def preprocess_image(image_path, target_size=(224, 224)):
    """Prepares an image for model inference."""
    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict(model, label_encoder, image_tensor):
    """Makes a prediction on a preprocessed image tensor."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = label_encoder.classes_[predicted_idx]
        
        # Create a dictionary of all class probabilities
        all_probs = {label_encoder.classes_[i]: prob.item() for i, prob in enumerate(probabilities)}

    return predicted_class, confidence.item(), all_probs

def main():
    """Main function for command-line inference."""
    parser = argparse.ArgumentParser(description='Breast Cancer Detection Inference (PyTorch)')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the mammogram image for prediction')
    # --- UPDATED: Default model path is now the v3 model ---
    parser.add_argument('--model', type=str, default='breast_cancer_model_v3.pth',
                        help='Path to the trained PyTorch model file (.pth)')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        return
    
    # Load model and encoder
    model, label_encoder = load_pytorch_model(args.model)
    if model is None:
        return
        
    # Preprocess image
    image_tensor = preprocess_image(args.image)
    if image_tensor is None:
        return
    
    # Make prediction
    print(f"\nAnalyzing image: {args.image}")
    predicted_class, confidence, all_probs = predict(model, label_encoder, image_tensor)
    
    # Print results
    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\n--- All Class Probabilities ---")
    # Sort probabilities for clearer presentation
    sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
    for class_name, prob in sorted_probs:
        print(f"  {class_name}: {prob:.2%}")

    print("\n--- Interpretation ---")
    if predicted_class == 'Normal':
        print("The model predicts this mammogram shows normal breast tissue.")
    elif predicted_class == 'Benign':
        print("The model predicts this mammogram shows benign (non-cancerous) findings.")
    elif predicted_class == 'Malignant':
        print("The model predicts this mammogram shows potentially malignant (cancerous) findings.")
    print("\nNote: This is an AI prediction and should not replace professional medical diagnosis.")


if __name__ == "__main__":
    main()
