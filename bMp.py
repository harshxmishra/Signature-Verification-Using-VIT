import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification
from PIL import Image
import os
from sklearn.metrics import accuracy_score, classification_report

# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 16
learning_rate = 1e-4
num_epochs = 10
validation_split = 0.1

# Custom dataset class for loading user signatures
class UserSignaturesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.users = sorted(os.listdir(root_dir))
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}
        self.images = []
        self.labels = []
        for user in self.users:
            user_dir = os.path.join(root_dir, user)
            for filename in os.listdir(user_dir):
                self.images.append(os.path.join(user_dir, filename))
                self.labels.append(self.user_to_idx[user])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load datasets
train_dataset = UserSignaturesDataset(root_dir='/home/test/Desktop/Split the data/train', transform=data_transform)
test_dataset = UserSignaturesDataset(root_dir='/home/test/Desktop/Split the data/test', transform=data_transform)
validation_dataset = UserSignaturesDataset(root_dir='/home/test/Desktop/Split the data/val', transform=data_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Initialize ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(train_dataset.users))
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)  # Adjust for ViTForImageClassification outputs

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.logits, labels)  # Validation loss

            val_loss += loss.item() * images.size(0)

            # Predictions
            _, preds = torch.max(outputs.logits, 1)  # Adjust for ViTForImageClassification outputs

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(validation_dataset)
    val_accuracy = accuracy_score(val_labels, val_preds)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save trained model
model_path = '/home/test/PycharmProjects/SignatureVerification/modeler.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Testing
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Predictions
        _, preds = torch.max(outputs.logits, 1)  # Adjust for ViTForImageClassification outputs

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(test_labels, test_preds, target_names=test_dataset.users))
