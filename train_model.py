import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30 
LEARNING_RATE = 1e-4 # Start with a slightly higher LR for fine-tuning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Custom Dataset Class ---
class MammogramDataset(Dataset):
    """A custom dataset class that loads images from paths specified in a DataFrame."""
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['ImageName']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['encoded_label']
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Model Definition ---
class BreastCancerModel(nn.Module):
    def __init__(self, num_classes):
        super(BreastCancerModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # --- IMPROVEMENT: Fine-tuning - Unfreeze last few layers ---
        # Freeze all layers first
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier and the last few convolutional blocks
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

# --- Plotting Functions ---
def plot_training_history(history):
    """Plots and saves the training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_v3.png')
    print("\nSaved training history plot to 'training_history_v3.png'")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_v3.png')
    print("Saved confusion matrix plot to 'confusion_matrix_v3.png'")
    plt.show()

# --- Training and Evaluation Loop ---
def train_model_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model_v3.pth')
            print(f"  -> Best model saved with validation accuracy: {best_val_accuracy:.2f}%")
    
    return history

def main():
    print("Step 1: Loading and preparing data...")
    df = pd.read_csv('dataset/final_dataset.csv')
    df = df[df['BI-RADS_Score'] != 'Not Found'].copy()
    
    def map_birads_to_label(score):
        score = int(score)
        if score in [1, 2]:
            return 'Normal'
        elif score == 3:
            return 'Benign'
        else: # Scores 4 and 5
            return 'Malignant'
    
    df['label'] = df['BI-RADS_Score'].apply(map_birads_to_label)
    print("Class distribution:\n", df['label'].value_counts())

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(df['label']),
        y=df['label'].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"\nCalculated Class Weights: {class_weights}")

    main_label_encoder = LabelEncoder()
    df['encoded_label'] = main_label_encoder.fit_transform(df['label'])

    print("\nStep 2: Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    print("\nStep 3: Creating datasets and dataloaders...")
    image_dir = 'dataset/images'
    
    # --- IMPROVEMENT: More aggressive data augmentation ---
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MammogramDataset(train_df, image_dir, transform=train_transform)
    val_dataset = MammogramDataset(val_df, image_dir, transform=val_test_transform)
    test_dataset = MammogramDataset(test_df, image_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nStep 4: Initializing model...")
    num_classes = len(main_label_encoder.classes_)
    model = BreastCancerModel(num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    print("\nStep 5: Starting model training...")
    history = train_model_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)
    plot_training_history(history)

    print("\nStep 6: Evaluating the best model on the test set...")
    model.load_state_dict(torch.load('best_model_v3.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nClassification Report on Test Set:")
    print(classification_report(all_labels, all_preds, target_names=main_label_encoder.classes_))
    plot_confusion_matrix(all_labels, all_preds, class_names=main_label_encoder.classes_)

    print("\nStep 7: Saving final model for the app...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': main_label_encoder,
    }, 'breast_cancer_model_v3.pth')
    print("Final model saved as 'breast_cancer_model_v3.pth'")

if __name__ == "__main__":
    main()
