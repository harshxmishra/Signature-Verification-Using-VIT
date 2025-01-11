import os
from PIL import Image  # For image processing
from sklearn.model_selection import train_test_split

# Define paths and parameters
data_dir = r"/home/test/Desktop/cropped1"  # Replace with your data directory
output_dir = r'/home/test/Desktop/splitdata' # Replace with your output directory
image_size = (256, 256)  # Target image size for normalization

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path):
  image = Image.open(image_path).convert('L')  # Convert to grayscale
  resized_image = image.resize(image_size, Image.NEAREST)  # Resize with antialiasing
  # You can add more preprocessing steps here as needed (e.g., normalization)
  # Normalize pixel values (example):
  # image_data = np.array(resized_image)
  # normalized_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
  return resized_image

# Loop through each user directory
for user_dir in os.listdir(data_dir):
  user_path = os.path.join(data_dir, user_dir)
  if os.path.isdir(user_path):
    images = []
    for filename in os.listdir(user_path):
      image_path = os.path.join(user_path, filename)
      if os.path.isfile(image_path) and filename.lower().endswith(".bmp"):  # Check for PNG extension
        preprocessed_image = preprocess_image(image_path)
        images.append(preprocessed_image)

    # Split images into train, validation, and test sets
    X_train, X_test_val, _, _ = train_test_split(images, range(len(images)), test_size=0.2, random_state=42)
    X_val, X_test = train_test_split(X_test_val, test_size=0.5, random_state=42)

    # Save preprocessed images to separate directories
    train_dir = os.path.join(output_dir, "train", user_dir)
    val_dir = os.path.join(output_dir, "val", user_dir)
    test_dir = os.path.join(output_dir, "test", user_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i, image in enumerate(X_train):
      image.save(os.path.join(train_dir, f"{i}.bmp"))

    for i, image in enumerate(X_val):
      image.save(os.path.join(val_dir, f"{i}.bmp"))

    for i, image in enumerate(X_test):
      image.save(os.path.join(test_dir, f"{i}.bmp"))

    print(f"Preprocessed and saved images for user: {user_dir}")

print("Preprocessing complete!")