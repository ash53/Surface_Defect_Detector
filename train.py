# ==============================================================================
# train.py
#
# This script trains a deep learning model to detect surface defects on
# manufacturing components using transfer learning with a pre-trained ResNet18.
#
# ==============================================================================

# --- 1. Import Necessary Libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the entire training and evaluation pipeline.
    """
    # --- 2. Configuration and Setup ---

    # Set the device to a GPU if available, otherwise use the CPU.
    # Training on a GPU is significantly faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path to dataset's root folder.
    data_dir = 'data'

    # Hyperparameters and model settings
    num_classes = 6      # Number of defect categories
    batch_size = 32      # Number of images to process in one go
    num_epochs = 25      # Number of times to loop over the entire training dataset
    learning_rate = 0.001 # Controls how much the model's weights are adjusted

    # --- 3. Data Loading and Augmentation ---

    # Define a set of image transformations for data preprocessing and augmentation.
    data_transforms = {
        # For the training set, we apply data augmentation to make the model more robust.
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),      # Randomly crop and resize images
            transforms.RandomHorizontalFlip(),      # Randomly flip images horizontally
            transforms.ToTensor(),                  # Convert images to PyTorch Tensors
            # Normalize images with the mean and standard deviation of the ImageNet dataset.
            # This is a crucial step for transfer learning with models pre-trained on ImageNet.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # For the validation set, we only apply the necessary preprocessing without random augmentation.
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets from the organized folders using ImageFolder.
    print("Loading datasets...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    # Create DataLoaders to serve the data to the model in batches.
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
                   
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")
    print(f"Training data size: {dataset_sizes['train']}")
    print(f"Validation data size: {dataset_sizes['val']}")

    # --- 4. Model Definition (Transfer Learning) ---

    # Load the ResNet18 model pre-trained on ImageNet.
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Freeze all the layers of the pre-trained model.
    # We do this to keep the learned features from ImageNet and only train the final layer.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer (the classifier) with a new one.
    # The new layer will be trained to classify our specific number of defect classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move the model to the configured device (GPU or CPU).
    model = model.to(device)
    print("Model architecture loaded and modified for transfer learning.")

    # --- 5. Training Setup ---

    # Define the loss function. CrossEntropyLoss is standard for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer. Adam is a popular and effective choice.
    # We only pass the parameters of the final, unfrozen layer (model.fc) to the optimizer.
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # --- 6. Execute Training ---
    print("\nStarting model training...")
    model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=num_epochs)

    # --- 7. Visualize Results ---
    print("\nTraining finished. Visualizing results...")
    plot_history(history)

    # --- 8. Save the Model ---
    # Create the directory if it doesn't exist
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        print("Created 'saved_models' directory.")
        
    # Save the trained model's weights for future use.
    torch.save(model.state_dict(), 'saved_models/surface_defect_detector_best.pth')
    print("\nBest model weights saved to 'saved_models/surface_defect_detector_best.pth'")


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    """
    Handles the training and validation loop for the model.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Store history for plotting
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Store results
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if we achieve a new best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def plot_history(history):
    """
    Visualizes the training and validation accuracy and loss over epochs.
    """
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(14, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Training and Validation History')
    plt.show()


if __name__ == '__main__':
    # This block ensures that the main() function is called only when
    # the script is executed directly (not when imported as a module).
    main()