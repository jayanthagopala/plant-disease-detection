"""Training script for plant disease detection model."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from models.cnn_model import create_model, save_model
from data.dataset import create_data_loaders
from data.utils import create_sample_dataset, analyze_dataset


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str
) -> Dict:
    """Evaluate the model on test set."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    
    # Classification report
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'labels': all_labels
    }


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Path
) -> None:
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path
) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train plant disease detection model')
    parser.add_argument('--data_dir', type=str, default='data/plant_diseases', 
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs', 
                       help='Path to output directory')
    parser.add_argument('--model_name', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                       help='Base model architecture')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, 
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset if data directory is empty')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data directory exists and create sample if needed
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        if args.create_sample:
            print("Creating sample dataset...")
            create_sample_dataset(data_dir, num_samples_per_class=50)
        else:
            print(f"Data directory {data_dir} is empty. Use --create_sample to create sample data.")
            return
    
    # Analyze dataset
    print("Analyzing dataset...")
    dataset_stats = analyze_dataset(data_dir)
    print(f"Dataset statistics: {dataset_stats}")
    
    # Get class names from directory structure
    class_names = [d.name for d in data_dir.iterdir() if d.is_dir()]
    class_names.sort()
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        data_dir=data_dir,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = create_model(
        num_classes=num_classes,
        model_name=args.model_name,
        device=device
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    print("Starting training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model=model,
                save_path=output_dir / "best_model.pth",
                epoch=epoch,
                loss=val_loss,
                accuracy=val_acc,
                class_names=class_names,
                model_name=args.model_name,
                num_classes=num_classes
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, class_names, device)
    
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    # Save final model
    save_model(
        model=model,
        save_path=output_dir / "final_model.pth",
        epoch=epoch,
        loss=val_loss,
        accuracy=test_results['accuracy'],
        class_names=class_names,
        model_name=args.model_name,
        num_classes=num_classes
    )
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        output_dir / "training_history.png"
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(test_results['confusion_matrix']),
        class_names,
        output_dir / "confusion_matrix.png"
    )
    
    # Save results
    results = {
        'model_name': args.model_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_results['accuracy']),
        'training_history': {
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'train_accuracies': [float(x) for x in train_accs],
            'val_accuracies': [float(x) for x in val_accs]
        },
        'test_results': {
            'accuracy': float(test_results['accuracy']),
            'predictions': [int(x) for x in test_results['predictions']],
            'labels': [int(x) for x in test_results['labels']],
            'confusion_matrix': test_results['confusion_matrix']
        }
    }
    
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {output_dir}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_results['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
