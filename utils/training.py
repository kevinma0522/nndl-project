import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class MultiLabelTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        super_preds = []
        super_targets = []
        sub_preds = []
        sub_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for images, super_labels, sub_labels in pbar:
            images = images.to(self.device)
            super_labels = super_labels.to(self.device)
            sub_labels = sub_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            super_output, sub_output = self.model(images)
            
            # Calculate loss
            super_loss = self.criterion(super_output, super_labels)
            sub_loss = self.criterion(sub_output, sub_labels)
            loss = super_loss + sub_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            super_preds.extend((super_output > 0.5).cpu().numpy())
            super_targets.extend(super_labels.cpu().numpy())
            sub_preds.extend((sub_output > 0.5).cpu().numpy())
            sub_targets.extend(sub_labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        # Calculate metrics
        metrics = self._calculate_metrics(
            super_preds, super_targets,
            sub_preds, sub_targets
        )
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        super_preds = []
        super_targets = []
        sub_preds = []
        sub_targets = []
        
        with torch.no_grad():
            for images, super_labels, sub_labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                super_labels = super_labels.to(self.device)
                sub_labels = sub_labels.to(self.device)
                
                # Forward pass
                super_output, sub_output = self.model(images)
                
                # Calculate loss
                super_loss = self.criterion(super_output, super_labels)
                sub_loss = self.criterion(sub_output, sub_labels)
                loss = super_loss + sub_loss
                
                # Update metrics
                total_loss += loss.item()
                super_preds.extend((super_output > 0.5).cpu().numpy())
                super_targets.extend(super_labels.cpu().numpy())
                sub_preds.extend((sub_output > 0.5).cpu().numpy())
                sub_targets.extend(sub_labels.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            super_preds, super_targets,
            sub_preds, sub_targets
        )
        metrics['loss'] = total_loss / len(val_loader)
        
        # Update learning rate
        self.scheduler.step(metrics['loss'])
        
        return metrics
    
    def _calculate_metrics(self, super_preds, super_targets, sub_preds, sub_targets):
        metrics = {}
        
        # Super-class metrics
        metrics['super_f1'] = f1_score(super_targets, super_preds, average='macro')
        metrics['super_precision'] = precision_score(super_targets, super_preds, average='macro')
        metrics['super_recall'] = recall_score(super_targets, super_preds, average='macro')
        
        # Sub-class metrics
        metrics['sub_f1'] = f1_score(sub_targets, sub_preds, average='macro')
        metrics['sub_precision'] = precision_score(sub_targets, sub_preds, average='macro')
        metrics['sub_recall'] = recall_score(sub_targets, sub_preds, average='macro')
        
        return metrics
    
    def save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics'] 