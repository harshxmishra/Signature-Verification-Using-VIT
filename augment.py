import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory containing user folders
base_dir = "/home/test/Desktop/cropped1"


# Function to augment images
def augment_images(user_dir, target_count, prefix):
    datagen = ImageDataGenerator(
        brightness_range=[0.5, 1.0],
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2
    )

    images = [os.path.join(user_dir, img) for img in os.listdir(user_dir) if img.endswith('.bmp')]
    current_count = len(images)

    if current_count >= target_count:
        return

    augment_count = target_count - current_count
    x = []

    for img_path in images:
        image = Image.open(img_path)
        image = image.convert('RGB')  # Ensure image is RGB (required for some augmentations)

        # Resize or pad the image to a common size (e.g., 224x224)
        image = image.resize((224, 224))  # Adjust dimensions as needed

        # Convert image to numpy array
        x.append(np.array(image))

    x = np.array(x)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=user_dir, save_prefix=prefix, save_format='bmp'):
        i += 1
        if i >= augment_count:
            break


# Iterate over each user folder
for user_folder in os.listdir(base_dir):
    user_folder_path = os.path.join(base_dir, user_folder)

    # Check if it's a directory
    if not os.path.isdir(user_folder_path):
        continue

    print(f"Augmenting images for user: {user_folder}...")

    # Perform augmentation for each user folder
    augment_images(user_folder_path, 1000, prefix='aug')

print('Data augmentation complete.')