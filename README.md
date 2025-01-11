# Signature Verification using Vision Transformer (ViT)

This project implements a signature verification system using a Vision Transformer (ViT) model. It leverages image augmentation, preprocessing, model training, and testing to classify signatures and identify the signer.

## Project Structure

The project consists of the following Python scripts:

- **`augment.py`**: This script performs image augmentation to increase the size and diversity of the training dataset. It reads images from user-specific directories and applies transformations such as brightness adjustment, horizontal flip, shear, and zoom. The augmented images are saved back into the respective user directories.

- **`bMp.py`**: This is the main script for training and evaluating the ViT model. It defines a custom dataset (`UserSignaturesDataset`) to load signature images and their corresponding labels. It uses a pre-trained ViT model (`google/vit-base-patch16-224-in21k`) and fine-tunes it on the signature data. The script includes a training loop with validation and saves the trained model weights to `modeler.pth`. It also evaluates the model on a test set and prints the accuracy and classification report.

- **`preprocess.py`**: This script handles the preprocessing of the signature images. It loads images, converts them to grayscale, resizes them, and splits them into training, validation, and testing sets. The preprocessed images are saved into separate directories for each set.

- **`testing.py`**: This script is used for testing the trained signature verification model. It loads the model weights from `modeler.pth` and preprocesses new signature images. It then uses the loaded model to predict the signer of the input signature and prints the predicted user name.

## Dependencies

The project uses the following Python libraries:

- `torch`
- `torch.nn`
- `torch.optim`
- `torch.utils.data`
- `torchvision`
- `transformers`
- `PIL (Pillow)`
- `os`
- `sklearn.metrics`
- `sklearn.model_selection`
- `tensorflow.keras`
- `numpy`

You can install the necessary dependencies using pip:

```bash
pip install torch torchvision transformers scikit-learn Pillow tensorflow numpy
```

## Data Preparation

1. Organize your signature images into a directory structure where each subdirectory represents a user, and the images within each subdirectory are the signatures of that user.
2. Update the `data_dir` variable in `preprocess.py` to point to your data directory.
3. Run `preprocess.py` to preprocess the images and split them into training, validation, and testing sets.

## Training the Model

1. Ensure that the paths in `bMp.py` for the training, testing, and validation datasets are correctly set.
2. Run `augment.py` to augment the training data.
3. Run `bMp.py` to train the ViT model. The trained model weights will be saved to `modeler.pth`.

## Testing the Model

1. Update the `model_path` variable in `testing.py` to the correct path of your saved model weights (`modeler.pth`).
2. Organize the signature images you want to test into a directory structure similar to the training data.
3. Update the `test_folder_path` variable in `testing.py` to point to your test data directory.
4. Ensure the `user_names` list in `testing.py` matches the order of users in your training data.
5. Run `testing.py` to test the model on the provided images.

## Model Architecture

The project utilizes the pre-trained `google/vit-base-patch16-224-in21k` Vision Transformer model from the `transformers` library. This model is fine-tuned on the signature data to perform signature classification.

## Usage

To use the signature verification system, follow these steps:

1. Prepare your signature data as described in the "Data Preparation" section.
2. Train the model using the `bMp.py` script.
3. Test the model using the `testing.py` script.

## Notes

- Adjust the hyperparameters in `bMp.py` (e.g., batch size, learning rate, number of epochs) as needed for optimal performance.
- Ensure that the paths to the data directories and model weights are correctly configured in the respective scripts.
- The `user_names` list in `testing.py` must correspond to the order of users in the training data to ensure correct mapping of predictions to user names.
