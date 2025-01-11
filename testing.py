import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import os

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Example preprocess_image function
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB mode
    image = data_transform(image)  # Apply the defined transformations
    return image.unsqueeze(0)  # Add batch dimension

# Initialize the model architecture (assuming the same configuration as during training)
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=16)  # Adjust num_labels as needed

# Specify the path to the .pth file containing the model weights
model_path = '/home/test/PycharmProjects/SignatureVerification/modeler.pth'  # Adjust based on your extracted path

# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Adjust map_location as needed

# Evaluate mode (important if your model has layers like Dropout or BatchNorm that behave differently during training and evaluation)
model.eval()

# Example: Assume you have a list of user names corresponding to class indices
user_names = ['Aditya', 'Ayushi', 'Deepak', 'Depaam', 'Deevesh', 'Harsh', 'Himarshini', 'Moksh', 'Nikita', 'Ninja', 'Nishant', 'Priyadarshini', 'Rajsabi', 'Rishab', 'Rudraksh', 'Shomesh']

# Path to the test folder containing subfolders for each user
test_folder_path = '/home/test/PycharmProjects/new_test'  # Adjust based on your folder structure

# Iterate through the test folder and perform inference on each image
for user_folder in os.listdir(test_folder_path):
    user_folder_path = os.path.join(test_folder_path, user_folder)
    if os.path.isdir(user_folder_path):
        print(f"Processing images for user: {user_folder}")
        for image_name in os.listdir(user_folder_path):
            image_path = os.path.join(user_folder_path, image_name)
            if os.path.isfile(image_path):
                input_image = preprocess_image(image_path)

                # Perform inference
                with torch.no_grad():
                    outputs = model(input_image)

                # Get predicted class index
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

                # Get predicted user name
                predicted_user_name = user_names[predicted_class_idx]

                print(f"Image: {image_name}, Predicted user name: {predicted_user_name}")
