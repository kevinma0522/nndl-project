# Multi-Label Image Classification: CNN vs MLP

This repository contains the implementation of a comparative study between a CNN-based classifier and an MLP for multi-label image classification. The project is based on the research proposal by Thomas Chen and Kevin Ma from Columbia University.

## Project Structure

```
.
├── models/
│   ├── cnn.py          # CNN model implementation
│   └── mlp.py          # MLP model implementation
├── utils/
│   ├── data.py         # Data loading and preprocessing
│   └── training.py     # Training utilities
├── train.py            # Main training script
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the models:

```bash
python train.py --model cnn  # For CNN model
python train.py --model mlp  # For MLP model
```

## Model Architectures

### CNN Model
- Inspired by AlexNet architecture
- 3-5 convolutional layers with BatchNorm and ReLU
- Max pooling layers
- Two fully connected layers
- Multi-label output with sigmoid activation

### MLP Model
- 2-4 hidden layers with ReLU and BatchNorm
- Input: Flattened 64x64x3 images
- Multi-label output with sigmoid activation

## Data

The models are designed to work with 64x64 RGB images in a multi-label classification setting, predicting both super-class and subclass labels.

## Training

Both models are trained using:
- Binary Cross Entropy Loss
- Adam Optimizer
- Learning Rate Scheduling
- Data Augmentation (random flips, crops, color jittering) 