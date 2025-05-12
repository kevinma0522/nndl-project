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

Prepare Data by running python3 prepare_data.py


To train the models:

```bash
python3 train.py \
  --model cnn \
  --data_dir ./dataset/images \
  --super_labels ./dataset/super_labels.npy \
  --sub_labels ./dataset/sub_labels.npy \
  --num_super_classes 20 \
  --num_sub_classes 100 \
  --batch_size 32 \
  --epochs 2  # For CNN model
python3 train.py \
  --model mlp \
  --data_dir ./dataset/images \
  --super_labels ./dataset/super_labels.npy \
  --sub_labels ./dataset/sub_labels.npy \
  --num_super_classes 20 \
  --num_sub_classes 100 \
  --batch_size 32 \
  --epochs 20  # For MLP model
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