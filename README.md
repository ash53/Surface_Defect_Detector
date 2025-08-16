# AI-Powered Surface Defect Detector

A deep learning model built with PyTorch to detect and classify common surface defects on manufacturing components. This project uses **transfer learning** from a pre-trained ResNet18 model to achieve high accuracy with a limited dataset.

---

## ✨ Features

- **Framework:** PyTorch
- **Model:** Pre-trained ResNet18
- **Technique:** Transfer Learning (Feature Extraction)
- **Dataset:** NEU Surface Defect Database
- **Accuracy:** Achieves >95% validation accuracy
- **Classes (6):** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches

---

## 📂 Project Structure

```
surface-defect-detector/
│
├── data/                 # Contains the training and validation images
│   ├── train/
│   └── val/
│
├── saved_models/         # Stores the final trained model weights (.pth files)
│
├── results/              # Stores output plots and logs
│
├── train.py              # The main script to run training
├── requirements.txt      # A list of Python dependencies
└── README.md             # You are here!
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.8+
- pip package manager

### 1. Set Up the Project

First, create the main project folder. You can clone this repository from Git, or create the structure manually.

### 2. Download and Organize the Dataset

This is the most important manual step.

1.  **Download** the dataset from the [NEU Surface Defect Database on Kaggle](https://www.kaggle.com/datasets/kaustubhb999/northeastern-university-neu-surface-defect).
2.  **Create** the `data/train` and `data/val` directories.
3.  **Split** the images from the downloaded dataset into the `train` and `val` folders. A good split is 80% for training and 20% for validation.
4.  **Ensure** the final structure inside the `data` folder matches the one described in the "Project Structure" section, with a subfolder for each defect type.

### 3. Install Dependencies

Navigate to the project's root directory in your terminal and install the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

1.  **Modify the data path:** Open the `train.py` script and update the `data_dir` variable to point to your `data` folder.

    ```python
    # In train.py, line ~19
    data_dir = 'path/to/your/data' # <-- CHANGE THIS
    ```

2.  **Start Training:** Run the following command from the project's root directory.

    ```bash
    python train.py
    ```

The script will start the training process and print the loss and accuracy for each epoch.

---

## 📈 Expected Results

The model will train for 25 epochs. After training is complete, you will see a summary of the best validation accuracy achieved. A plot showing the training and validation accuracy/loss over all epochs will be displayed.
