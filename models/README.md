# Language Detection Project

Add this directory to your .gitignore to avoid committing large model files to git.

## Saved Models

This directory contains trained model checkpoints:

### Classical ML Models
- `svm_linear.pkl` - Linear Support Vector Machine
- `svm_rbf.pkl` - SVM with RBF kernel (optional)
- `random_forest.pkl` - Random Forest Classifier
- `logistic_regression.pkl` - Logistic Regression
- `naive_bayes.pkl` - Multinomial Naive Bayes

### Deep Learning Models
- `cnn_model.pth` - Character-level CNN model
- `bilstm_model.pth` - Bidirectional LSTM model
- `hybrid_model.pth` - Hybrid CNN-LSTM model
- `best_model.pth` - Best performing model checkpoint
- `char_to_idx.pkl` - Character vocabulary mapping

## Loading Saved Models

### Classical ML
```python
import pickle

with open('models/saved_models/svm_linear.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
```

### Deep Learning
```python
import torch
import pickle

# Load vocabulary
with open('models/saved_models/char_to_idx.pkl', 'rb') as f:
    char_to_idx = pickle.load(f)

# Load model
from src.deep_learning import CharCNN

model = CharCNN(vocab_size=len(char_to_idx), embedding_dim=256, num_classes=num_classes)
model.load_state_dict(torch.load('models/saved_models/cnn_model.pth'))
model.eval()

# Make predictions
# ... (see src/deep_learning.py for full example)
```

## .gitignore

Add the following to your .gitignore:
```
models/saved_models/*.pkl
models/saved_models/*.pth
```
