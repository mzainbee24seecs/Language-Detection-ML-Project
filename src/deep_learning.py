"""
Deep Learning Models for Language Detection using PyTorch
==========================================================
This module implements neural network architectures for language detection:
- Character-level CNN
- BiLSTM Network
- Hybrid CNN-LSTM Architecture

Author: [Your Name]
Course: CS 470 - Machine Learning
Project: Language Detection (Multi-class Classification)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class TextDataset(Dataset):
    """
    Custom Dataset for text data.
    """
    
    def __init__(self, texts: np.ndarray, labels: np.ndarray, 
                 char_to_idx: Dict, max_length: int = 200):
        """
        Initialize the dataset.
        
        Args:
            texts: Array of text strings
            labels: Array of integer labels
            char_to_idx: Dictionary mapping characters to indices
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) 
                  for char in text[:self.max_length]]
        
        # Pad sequence
        if len(indices) < self.max_length:
            indices += [self.char_to_idx['<PAD>']] * (self.max_length - len(indices))
        
        return torch.LongTensor(indices), torch.LongTensor([label])


def build_vocab(texts: np.ndarray, min_freq: int = 2) -> Tuple[Dict, Dict]:
    """
    Build character vocabulary from texts.
    
    Args:
        texts: Array of text strings
        min_freq: Minimum frequency for a character to be included
        
    Returns:
        Tuple of (char_to_idx, idx_to_char) dictionaries
    """
    print("Building vocabulary...")
    char_freq = {}
    
    for text in texts:
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
    
    # Special tokens
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for char, freq in sorted(char_freq.items()):
        if freq >= min_freq:
            char_to_idx[char] = idx
            idx += 1
    
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    return char_to_idx, idx_to_char


class CharCNN(nn.Module):
    """
    Character-level CNN for text classification.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int,
                 num_filters: int = 128, kernel_sizes: List[int] = [3, 4, 5],
                 dropout: float = 0.5):
        """
        Initialize the CharCNN model.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            num_classes: Number of output classes
            num_filters: Number of filters for each kernel size
            kernel_sizes: List of kernel sizes for convolutions
            dropout: Dropout probability
        """
        super(CharCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_length]
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, seq_length - kernel_size + 1]
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]
        
        # Apply dropout and fully connected layer
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)  # [batch_size, num_classes]
        
        return output


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for text classification.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize the BiLSTM model.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_classes: Number of output classes
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the final hidden states from both directions
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        concatenated = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # Apply dropout and fully connected layer
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)  # [batch_size, num_classes]
        
        return output


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture for text classification.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int,
                 kernel_size: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        """
        Initialize the Hybrid CNN-LSTM model.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            num_filters: Number of CNN filters
            kernel_size: CNN kernel size
            hidden_dim: LSTM hidden dimension
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(HybridCNNLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=1)
        self.lstm = nn.LSTM(num_filters, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_length]
        
        # CNN
        conv_out = torch.relu(self.conv(embedded))  # [batch_size, num_filters, seq_length]
        conv_out = conv_out.permute(0, 2, 1)  # [batch_size, seq_length, num_filters]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(conv_out)
        
        # Use final hidden states
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        concatenated = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # Fully connected
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        
        return output


class DeepLearningTrainer:
    """
    Trainer class for deep learning models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Using device: {self.device}")
        
    def train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int, learning_rate: float = 0.001,
             patience: int = 5) -> Dict:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            
        Returns:
            Dictionary containing training history
        """
        print("\n" + "="*60)
        print("TRAINING DEEP LEARNING MODEL")
        print("="*60)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/saved_models/best_model.pth')
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        
        # Load best model
        self.model.load_state_dict(torch.load('models/saved_models/best_model.pth'))
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'training_time': training_time,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Example usage of deep learning models.
    """
    set_seed(42)
    
    # Load text data
    print("Loading data...")
    with open('data/processed/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/processed/X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open('data/processed/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/processed/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/processed/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    with open('data/processed/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    # Build vocabulary
    char_to_idx, idx_to_char = build_vocab(X_train)
    num_classes = len(np.unique(y_train))
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, char_to_idx, max_length=200)
    val_dataset = TextDataset(X_val, y_val, char_to_idx, max_length=200)
    test_dataset = TextDataset(X_test, y_test, char_to_idx, max_length=200)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize CharCNN model
    model = CharCNN(
        vocab_size=len(char_to_idx),
        embedding_dim=256,
        num_classes=num_classes,
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        dropout=0.5
    )
    
    # Train model
    trainer = DeepLearningTrainer(model)
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=50, learning_rate=0.001, patience=5
    )
    
    # Plot training history
    trainer.plot_training_history('results/figures/training_history.png')
    
    # Save vocabulary
    with open('models/saved_models/char_to_idx.pkl', 'wb') as f:
        pickle.dump(char_to_idx, f)
    
    print("\nDeep learning model training completed!")


if __name__ == "__main__":
    main()
