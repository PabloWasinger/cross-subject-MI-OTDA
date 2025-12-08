"""
Training utilities for EEGNet and other PyTorch models.

Includes:
- Data loading and dataset handling
- Training loop with early stopping
- Fine-tuning support
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data stored as numpy arrays."""
    
    def __init__(self, X, y, transform=None):
        """
        Parameters
        ----------
        X : ndarray
            EEG data of shape (n_trials, n_channels, n_samples).
        y : ndarray
            Labels of shape (n_trials,).
        transform : callable, optional
            Transform to apply to each sample.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        # Add channel dimension for Conv2d: (Chans, Samples) -> (1, Chans, Samples)
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.tensor(y).long()

        return x, y


def create_dataloaders(X_train, y_train, X_val=None, y_val=None, 
                       batch_size=32, num_workers=0, transform=None):
    """
    Create PyTorch DataLoader objects from numpy arrays.
    
    Parameters
    ----------
    X_train : ndarray
        Training data, shape (n_trials, n_channels, n_samples).
    y_train : ndarray
        Training labels.
    X_val : ndarray, optional
        Validation data.
    y_val : ndarray, optional
        Validation labels.
    batch_size : int
        Batch size for training.
    num_workers : int
        Number of workers for data loading.
    transform : callable, optional
        Transform to apply.
    
    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader or None
    """
    train_dataset = EEGDataset(X_train, y_train, transform=transform)
    
    use_cuda = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True if len(train_dataset) > batch_size else False
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = EEGDataset(X_val, y_val, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda
        )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, apply_max_norm=True):
    """
    Perform one training epoch.
    
    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function.
    optimizer : Optimizer
        Optimizer.
    device : torch.device
        Device to use.
    apply_max_norm : bool
        Whether to apply max_norm constraint after each batch.
    
    Returns
    -------
    epoch_loss : float
    epoch_acc : float
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Apply max_norm constraint if model has it
        if apply_max_norm and hasattr(model, '_apply_max_norm'):
            model._apply_max_norm()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set.
    
    Parameters
    ----------
    model : nn.Module
        Model to validate.
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device to use.
    
    Returns
    -------
    val_loss : float
    val_acc : float
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_model(model, train_loader, val_loader=None, epochs=100, lr=1e-3,
                patience=10, save_path=None, verbose=True):
    """
    Train a PyTorch model with early stopping and checkpointing.
    
    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader, optional
        Validation data loader for early stopping.
    epochs : int
        Maximum number of epochs.
    lr : float
        Learning rate.
    patience : int
        Early stopping patience.
    save_path : str, optional
        Path to save best model checkpoint.
    verbose : bool
        Print training progress.
    
    Returns
    -------
    model : nn.Module
        Trained model.
    history : dict
        Training history with keys: train_loss, train_acc, val_loss, val_acc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    
    for epoch in iterator:
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break
            
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.1f}%'
                })
        else:
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'train_acc': f'{train_acc:.1f}%'
                })
    
    # Load best model if saved
    if save_path and val_loader is not None:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def finetune_model(model, train_loader, val_loader=None, epochs=50, lr=1e-4,
                   freeze_temporal=True, patience=10, save_path=None, verbose=True):
    """
    Fine-tune a pre-trained model on new data.
    
    Parameters
    ----------
    model : nn.Module
        Pre-trained model to fine-tune.
    train_loader : DataLoader
        Fine-tuning data loader.
    val_loader : DataLoader, optional
        Validation data loader.
    epochs : int
        Maximum number of fine-tuning epochs.
    lr : float
        Learning rate (typically lower than initial training).
    freeze_temporal : bool
        If True, freeze temporal filtering layers (conv1, bn1).
    patience : int
        Early stopping patience.
    save_path : str, optional
        Path to save fine-tuned model.
    verbose : bool
        Print progress.
    
    Returns
    -------
    model : nn.Module
        Fine-tuned model.
    history : dict
        Fine-tuning history.
    """
    # Freeze layers if requested
    if freeze_temporal and hasattr(model, 'freeze_temporal_layers'):
        model.freeze_temporal_layers()
        if verbose:
            print("Temporal layers frozen for fine-tuning")
    
    # Train with lower learning rate
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, patience=patience,
        save_path=save_path, verbose=verbose
    )
    
    return model, history


def predict(model, data_loader, device=None):
    """
    Get predictions from a trained model.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    data_loader : DataLoader
        Data loader.
    device : torch.device, optional
        Device to use.
    
    Returns
    -------
    predictions : ndarray
        Predicted labels.
    probabilities : ndarray
        Prediction probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)


def extract_features(model, data_loader, device=None):
    """
    Extract features from a trained model (for BOTDA-GL).
    
    Parameters
    ----------
    model : nn.Module
        Trained model with extract_features method.
    data_loader : DataLoader
        Data loader.
    device : torch.device, optional
        Device to use.
    
    Returns
    -------
    features : ndarray
        Extracted features, shape (n_samples, feature_dim).
    labels : ndarray
        Corresponding labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            
            if hasattr(model, 'extract_features'):
                features = model.extract_features(data)
            else:
                # Fallback: use forward hooks or manual extraction
                raise NotImplementedError("Model must have extract_features method")
            
            all_features.extend(features.cpu().numpy())
            all_labels.extend(target.numpy())
    
    return np.array(all_features), np.array(all_labels)


def load_checkpoint(model, checkpoint_path, device=None):
    """
    Load model from checkpoint.
    
    Parameters
    ----------
    model : nn.Module
        Model architecture (must match checkpoint).
    checkpoint_path : str
        Path to checkpoint file.
    device : torch.device, optional
        Device to load to.
    
    Returns
    -------
    model : nn.Module
        Model with loaded weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    return model

