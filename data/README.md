# Dataset Information

## Dataset Source

This directory should contain the language detection dataset.

### Recommended Datasets:

1. **Language Detection Dataset (Kaggle)**
   - Link: https://www.kaggle.com/datasets/basilb2s/language-detection
   - Contains: 10,337 samples across 17 languages
   - Format: CSV with 'Text' and 'Language' columns

2. **WiLI-2018 Dataset**
   - Link: https://zenodo.org/record/841984
   - Contains: 235,000 paragraphs from 235 languages
   - Format: CSV

3. **Language Identification Dataset**
   - Link: UCI ML Repository
   - Various text samples in multiple languages

## Directory Structure

```
data/
├── raw/
│   └── language_detection.csv    # Original dataset (place here)
├── processed/
│   ├── X_train.pkl               # Training text data
│   ├── X_val.pkl                 # Validation text data
│   ├── X_test.pkl                # Test text data
│   ├── y_train.pkl               # Training labels
│   ├── y_val.pkl                 # Validation labels
│   ├── y_test.pkl                # Test labels
│   ├── X_train_char.pkl          # Character TF-IDF features (train)
│   ├── X_val_char.pkl            # Character TF-IDF features (val)
│   ├── X_test_char.pkl           # Character TF-IDF features (test)
│   ├── X_train_word.pkl          # Word TF-IDF features (train)
│   ├── X_val_word.pkl            # Word TF-IDF features (val)
│   ├── X_test_word.pkl           # Word TF-IDF features (test)
│   ├── label_encoder.pkl         # Label encoder
│   ├── tfidf_char.pkl            # Character TF-IDF vectorizer
│   └── tfidf_word.pkl            # Word TF-IDF vectorizer
└── README.md                     # This file
```

## Dataset Format

The dataset should be a CSV file with at least two columns:
- **Text**: The text sample in various languages
- **Language**: The language label (e.g., 'English', 'Spanish', 'French', etc.)

Example:
```csv
Text,Language
"This is an example sentence in English","English"
"Esta es una oración de ejemplo en español","Spanish"
"Ceci est un exemple de phrase en français","French"
```

## Download Instructions

1. Download the dataset from one of the sources above
2. Place the CSV file in the `data/raw/` directory
3. Rename it to `language_detection.csv` (or update the path in main.py)
4. Run preprocessing: `python main.py --preprocess`

## Notes

- Ensure the dataset has sufficient samples for each language (at least 50-100 samples per language recommended)
- The preprocessing script will automatically handle data cleaning and splitting
- Character encoding should be UTF-8 to support multiple languages
