import argparse
import torch
import os
from models.cnn import CNNClassifier
from models.mlp import MLPClassifier
from utils.data import create_data_loaders
from utils.training import MultiLabelTrainer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train multi-label image classifier')
    parser.add_argument('--model', type=str, choices=['cnn', 'mlp'], required=True,
                      help='Model architecture to use')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the images')
    parser.add_argument('--super_labels', type=str, required=True,
                      help='Path to super-class labels numpy file')
    parser.add_argument('--sub_labels', type=str, required=True,
                      help='Path to sub-class labels numpy file')
    parser.add_argument('--num_super_classes', type=int, required=True,
                      help='Number of super-classes')
    parser.add_argument('--num_sub_classes', type=int, required=True,
                      help='Number of sub-classes')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load labels
    super_labels = np.load(args.super_labels)
    sub_labels = np.load(args.sub_labels)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        super_labels,
        sub_labels,
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model == 'cnn':
        model = CNNClassifier(args.num_super_classes, args.num_sub_classes)
    else:
        model = MLPClassifier(args.num_super_classes, args.num_sub_classes)
    
    model = model.to(device)
    
    # Create trainer
    trainer = MultiLabelTrainer(model, device, learning_rate=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print('Training metrics:')
        for k, v in train_metrics.items():
            print(f'{k}: {v:.4f}')
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print('Validation metrics:')
        for k, v in val_metrics.items():
            print(f'{k}: {v:.4f}')
        
        # Save checkpoint if validation loss improved
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'{args.model}_best.pth'
            )
            trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
            print(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Save latest checkpoint
        latest_path = os.path.join(
            args.checkpoint_dir,
            f'{args.model}_latest.pth'
        )
        trainer.save_checkpoint(latest_path, epoch, val_metrics)

if __name__ == '__main__':
    main() 