# Language Detection - Multi-class Text Classification

**CS 470: Machine Learning - Fall 2025**  
**Members:** Muhammad Zain, Mubashir Qurashi 
**CMS ID:**502138
**Project ID:** 24 - Language Detection  
**Difficulty:** Easy  
**Instructor:** Dr. Sajjad Hussain  
**Institution:** SEECS, NUST Pakistan

---

## Abstract

This project implements a comprehensive language detection system using both classical machine learning and deep learning approaches. The goal is to classify text samples into their respective languages using multi-class classification techniques. We compare the performance of classical ML models (Support Vector Machines, Random Forest, Logistic Regression, Naive Bayes) against deep learning architectures (Character-level CNN using PyTorch) on a diverse multilingual text dataset. Our results demonstrate that Linear SVM achieves the best performance with an accuracy of 98.58%, closely followed by Logistic Regression at 98.52%. The Character-level CNN achieved 98.00% accuracy, demonstrating that classical ML approaches with proper feature engineering (character-level TF-IDF) can be highly competitive with deep learning for language detection tasks.

---

## 1. Introduction

### 1.1 Problem Statement
Language detection is a fundamental task in Natural Language Processing (NLP) that involves automatically identifying the language of a given text. This has numerous real-world applications including:
- Content filtering and routing in multilingual systems
- Machine translation preprocessing
- International customer support systems
- Social media analysis across different regions
- Document classification and indexing

### 1.2 Objectives
- Implement and compare at least 2 classical machine learning algorithms
- Design and train deep neural networks using PyTorch
- Conduct comprehensive model evaluation using multiple metrics
- Perform statistical comparison between classical and deep learning approaches
- Provide insights into which approach works best for language detection

---

## 2. Dataset Description

### 2.1 Dataset Source
**Dataset:** Language Detection Dataset  
**Source:** Kaggle - https://www.kaggle.com/datasets/basilb2s/language-detection  
**Format:** CSV file with text samples and language labels

### 2.2 Dataset Characteristics
- **Total Samples:** 10,337 text samples
- **Number of Languages:** 17 languages
- **Languages Included:** English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Arabic, Chinese, Japanese, Korean, Hindi, Turkish, Swedish, Danish, Norwegian
- **Features:** Raw text samples (sentences/phrases)
- **Target Variable:** Language label (multi-class classification)
- **Data Quality:** Clean, balanced dataset with no missing values

### 2.3 Data Preprocessing
1. **Text Cleaning:**
   - Lowercasing for consistency
   - Removal of extra whitespace
   - Removal of URLs and email addresses
   - Preservation of language-specific characters

2. **Feature Extraction:**
   - **Classical ML:** TF-IDF vectorization
     - Character n-grams (2-5 grams) - 10,000 features
     - Word n-grams (1-3 grams) - 5,000 features
   - **Deep Learning:** Character-level embeddings (256-dimensional)

3. **Data Split:**
   - Training: 70% (7,236 samples)
   - Validation: 15% (1,551 samples)
   - Testing: 15% (1,550 samples)
   - Stratified split to maintain class balance

---

## 3. Methodology

### 3.1 Classical Machine Learning Approaches

#### 3.1.1 Support Vector Machine (SVM)
- **Feature Representation:** TF-IDF with character n-grams (2-5 grams)
- **Kernel:** Linear and RBF kernels
- **Hyperparameter Tuning:**
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 'auto']
- **Cross-Validation:** 5-fold stratified CV

#### 3.1.2 Random Forest Classifier
- **Feature Representation:** TF-IDF with word n-grams (1-3 grams)
- **Hyperparameter Tuning:**
  - n_estimators: [100, 200, 300]
  - max_depth: [10, 20, 30, None]
  - min_samples_split: [2, 5, 10]
- **Cross-Validation:** 5-fold stratified CV

#### 3.1.3 Additional Models (Optional)
- Logistic Regression
- Naive Bayes (MultinomialNB)
- XGBoost

### 3.2 Deep Learning Approaches

#### 3.2.1 Character-level CNN Architecture
```
Input (seq_length) → 
Embedding Layer (vocab_size → 256) → 
Conv1D Layers (kernel_sizes: 3, 4, 5, filters: 128 each) → 
MaxPooling → 
Concatenate → 
Dropout(0.5) → 
Fully Connected (128 → num_classes) → 
Softmax
```

**Architecture Details:**
- Multiple parallel convolutions with different kernel sizes capture n-gram patterns
- Character-level embeddings learn language-specific character representations
- Dropout prevents overfitting
- Adam optimizer with learning rate scheduling

#### 3.2.2 Training Configuration
- **Framework:** PyTorch
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Batch Size:** 64
- **Epochs:** 50 (with early stopping)
- **Regularization:** Dropout (0.3-0.5), L2 regularization
- **Learning Rate Scheduler:** ReduceLROnPlateau

### 3.3 Hyperparameter Tuning Strategy
- Grid Search for Classical ML models
- Random Search / Manual tuning for Deep Learning models
- Validation set used for hyperparameter selection
- All models evaluated on held-out test set

---

## 4. Results & Analysis

### 4.1 Performance Comparison

#### Table 1: Model Performance on Test Set

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Key Features |
|-------|----------|-------------------|----------------|------------------|---------------|
| SVM (Linear) | **98.58%** | 0.9889 | 0.9876 | 0.9882 | Best overall performance |
| Logistic Regression | 98.52% | 0.9880 | 0.9869 | 0.9874 | Fast training, highly competitive |
| CNN (Char-level) | 98.00% | 0.9798 | 0.9819 | 0.9808 | Deep learning approach |
| Naive Bayes | 97.29% | 0.9825 | 0.9688 | 0.9739 | Fastest training time |
| Random Forest | 97.03% | 0.9753 | 0.9729 | 0.9738 | Good interpretability |

### 4.2 Confusion Matrix Analysis
Confusion matrices for all models show excellent classification performance with minimal misclassifications. The matrices are available in `results/figures/`:
- `cm_svm_linear.png` - Linear SVM confusion matrix
- `cm_logistic_regression.png` - Logistic Regression confusion matrix
- `cm_cnn.png` - CNN confusion matrix
- `cm_naive_bayes.png` - Naive Bayes confusion matrix
- `cm_random_forest.png` - Random Forest confusion matrix

All models show high diagonal values (correct predictions) and very few off-diagonal values (misclassifications), indicating robust performance across all language classes.

### 4.3 Visualizations
The following visualizations are available in `results/figures/`:
- **Training History:** `training_history_cnn.png` - Loss and accuracy curves for the CNN model showing convergence
- **ROC Curves:** `roc_cnn.png` - One-vs-rest ROC curves for multi-class classification
- **Model Comparison:** `model_comparison.png` - Comparative bar charts of model performance
- **Confusion Matrices:** See section 4.2 above

### 4.4 Statistical Significance Testing
Statistical comparison between models can be performed using McNemar's test (implemented in `src/evaluation.py`). The small performance differences between top models (SVM: 98.58%, Logistic Regression: 98.52%) suggest both are viable choices, with the selection depending on specific deployment requirements.

### 4.5 Key Findings
1. **Best Model:** Linear SVM achieved the highest accuracy of 98.58% with character-level TF-IDF features (2-5 grams)
2. **Classical vs Deep Learning:** Classical ML models with proper feature engineering (character n-grams) outperformed the deep learning CNN model, achieving 98.58% vs 98.00% accuracy
3. **Feature Importance:** Character n-grams (2-5 grams) proved highly effective for language detection, capturing language-specific character patterns
4. **Model Comparison:** All models achieved >97% accuracy, indicating language detection is well-suited for both classical and deep learning approaches
5. **Training Efficiency:** Classical models trained significantly faster (minutes) compared to deep learning (10-30 minutes), while achieving better or comparable performance
6. **Practical Value:** Linear SVM and Logistic Regression offer the best balance of accuracy, training speed, and model simplicity for production deployment

---

## 5. Conclusion & Future Work

### 5.1 Conclusion
This project successfully implemented and compared multiple approaches for language detection. Key takeaways include:
- **Linear SVM with character-level TF-IDF (2-5 grams) achieved the best performance at 98.58% accuracy**, demonstrating that classical ML with proper feature engineering remains highly competitive
- All five models (SVM, Logistic Regression, CNN, Naive Bayes, Random Forest) achieved >97% accuracy, indicating language detection is a well-defined problem with multiple viable solutions
- Classical ML approaches offer significant advantages in training efficiency and model interpretability while maintaining excellent performance
- Character-level features capture language-specific patterns more effectively than word-level features
- For production deployment, Linear SVM or Logistic Regression are recommended due to their optimal balance of accuracy, speed, and simplicity

### 5.2 Future Work
- Experiment with transformer-based models (BERT, XLM-RoBERTa)
- Extend to more languages and dialects
- Handle code-switched text (mixed languages)
- Implement real-time language detection API
- Explore few-shot learning for rare languages

---

## 6. Repository Structure

```
language-detection/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Preprocessed data
│   └── README.md              # Data documentation
│
├── src/
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── classical_ml.py         # Classical ML models
│   ├── deep_learning.py        # Deep learning models
│   ├── evaluation.py           # Evaluation metrics and visualization
│   └── utils.py                # Utility functions
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_Classical_ML.ipynb  # Classical ML experiments
│   └── 03_Deep_Learning.ipynb # Deep learning experiments
│
├── models/
│   └── saved_models/           # Trained model checkpoints
│
├── results/
│   ├── figures/                # Plots and visualizations
│   └── metrics/                # Performance metrics (CSV/JSON)
│
├── requirements.txt            # Python dependencies
├── main.py                     # Main pipeline script
└── README.md                   # This file
```

---

## 7. How to Run

### 7.1 Environment Setup
```bash
# Clone the repository
git clone [your-repo-url]
cd language-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Data Preparation
```bash
# Download dataset (instructions in data/README.md)
python src/data_preprocessing.py
```

### 7.3 Train Models
```bash
# Train all models
python main.py --train-all

# Train specific model
python main.py --model svm
python main.py --model cnn
```

### 7.4 Evaluate Models
```bash
python main.py --evaluate
```

---

## 8. Requirements

Key dependencies (see `requirements.txt` for versions):
- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+
- matplotlib 3.7+
- seaborn 0.12+
- scipy 1.11+
- joblib 1.3+

---

## 9. References

1. Cavnar, W. B., & Trenkle, J. M. (1994). "N-gram-based text categorization." Proceedings of SDAIR-94, 3rd Annual Symposium on Document Analysis and Information Retrieval.
2. Basilakis, B. (2018). "Language Detection Dataset." Kaggle. https://www.kaggle.com/datasets/basilb2s/language-detection
3. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, 2825-2830.
4. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS.
5. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification." EMNLP.
6. Scikit-learn Documentation. https://scikit-learn.org/
7. PyTorch Documentation. https://pytorch.org/docs/

---

## 10. Acknowledgments

- **Course Instructor:** Dr. Sajjad Hussain
- **Institution:** Department of Electrical and Computer Engineering, SEECS, NUST
- **Dataset:** Language Detection Dataset by Basilakis (Kaggle)
- **Framework:** PyTorch, scikit-learn

---
