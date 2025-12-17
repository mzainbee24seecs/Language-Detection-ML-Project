"""
Data Preprocessing Module for Language Detection
=================================================
This module handles loading, cleaning, and preprocessing text data for language detection.
Includes functions for feature extraction (TF-IDF) for classical ML models.

Author: [Your Name]
Course: CS 470 - Machine Learning
Project: Language Detection (Multi-class Classification)
"""

import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class LanguageDataProcessor:
    """
    Class for preprocessing language detection dataset.
    """
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the dataset CSV file
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.tfidf_char = None  # Character-level TF-IDF
        self.tfidf_word = None  # Word-level TF-IDF
        
    def load_data(self, csv_path: str = None) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file (optional, uses self.data_path if not provided)
            
        Returns:
            DataFrame containing the dataset
        """
        if csv_path is None:
            csv_path = self.data_path
            
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def explore_data(self, df: pd.DataFrame, text_col: str = 'Text', 
                    label_col: str = 'Language') -> Dict:
        """
        Perform exploratory data analysis.
        
        Args:
            df: DataFrame containing the data
            text_col: Name of the text column
            label_col: Name of the label column
            
        Returns:
            Dictionary containing EDA statistics
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print(f"\nDataset Shape: {df.shape}")
        print(f"Number of samples: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        
        # Check for missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Language distribution
        print(f"\nLanguage Distribution:")
        lang_dist = df[label_col].value_counts()
        print(lang_dist)
        
        # Text length statistics
        df['text_length'] = df[text_col].str.len()
        print(f"\nText Length Statistics:")
        print(df['text_length'].describe())
        
        # Average text length per language
        print(f"\nAverage Text Length per Language:")
        avg_length = df.groupby(label_col)['text_length'].mean().sort_values(ascending=False)
        print(avg_length)
        
        stats = {
            'n_samples': len(df),
            'n_languages': df[label_col].nunique(),
            'languages': df[label_col].unique().tolist(),
            'language_counts': lang_dist.to_dict(),
            'avg_text_length': df['text_length'].mean(),
            'min_text_length': df['text_length'].min(),
            'max_text_length': df['text_length'].max()
        }
        
        return stats
    
    def clean_text(self, text: str, remove_digits: bool = False, 
                   lowercase: bool = True) -> str:
        """
        Clean text data.
        
        Args:
            text: Input text string
            remove_digits: Whether to remove digits
            lowercase: Whether to convert to lowercase
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optionally remove digits
        if remove_digits:
            text = re.sub(r'\d+', '', text)
        
        return text
    
    def prepare_data(self, df: pd.DataFrame, text_col: str = 'Text',
                    label_col: str = 'Language', test_size: float = 0.15,
                    val_size: float = 0.15, clean: bool = True) -> Tuple:
        """
        Prepare data for modeling: clean, encode labels, and split.
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            clean: Whether to clean the text
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "="*60)
        print("PREPARING DATA")
        print("="*60)
        
        # Create a copy
        df = df.copy()
        
        # Clean text if requested
        if clean:
            print("Cleaning text data...")
            df['cleaned_text'] = df[text_col].apply(self.clean_text)
            text_col = 'cleaned_text'
        
        # Remove any rows with empty text
        df = df[df[text_col].str.len() > 0]
        print(f"Samples after cleaning: {len(df)}")
        
        # Encode labels
        print("Encoding labels...")
        y = self.label_encoder.fit_transform(df[label_col])
        X = df[text_col].values
        
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Split data: first split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"\nData Split:")
        print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_tfidf_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                             X_test: np.ndarray, 
                             analyzer: str = 'char',
                             ngram_range: Tuple[int, int] = (2, 5),
                             max_features: int = 10000) -> Tuple:
        """
        Create TF-IDF features for classical ML models.
        
        Args:
            X_train: Training text data
            X_val: Validation text data
            X_test: Test text data
            analyzer: 'char' or 'word'
            ngram_range: Range of n-grams
            max_features: Maximum number of features
            
        Returns:
            Tuple of (X_train_tfidf, X_val_tfidf, X_test_tfidf)
        """
        print("\n" + "="*60)
        print(f"CREATING TF-IDF FEATURES ({analyzer}-level)")
        print("="*60)
        
        print(f"N-gram range: {ngram_range}")
        print(f"Max features: {max_features}")
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit on training data and transform
        print("Fitting TF-IDF vectorizer on training data...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        print("Transforming validation and test data...")
        X_val_tfidf = vectorizer.transform(X_val)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Store the vectorizer
        if analyzer == 'char':
            self.tfidf_char = vectorizer
        else:
            self.tfidf_word = vectorizer
        
        print(f"\nTF-IDF Feature Shapes:")
        print(f"  Training: {X_train_tfidf.shape}")
        print(f"  Validation: {X_val_tfidf.shape}")
        print(f"  Test: {X_test_tfidf.shape}")
        print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf
    
    def save_processed_data(self, output_dir: str, **data_dict):
        """
        Save processed data and preprocessors.
        
        Args:
            output_dir: Directory to save files
            **data_dict: Keyword arguments containing data to save
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Save data arrays
        for name, data in data_dict.items():
            file_path = output_path / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {name} to {file_path}")
        
        # Save label encoder
        with open(output_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Saved label_encoder to {output_path / 'label_encoder.pkl'}")
        
        # Save TF-IDF vectorizers if they exist
        if self.tfidf_char is not None:
            with open(output_path / 'tfidf_char.pkl', 'wb') as f:
                pickle.dump(self.tfidf_char, f)
            print(f"Saved tfidf_char to {output_path / 'tfidf_char.pkl'}")
        
        if self.tfidf_word is not None:
            with open(output_path / 'tfidf_word.pkl', 'wb') as f:
                pickle.dump(self.tfidf_word, f)
            print(f"Saved tfidf_word to {output_path / 'tfidf_word.pkl'}")
    
    def load_processed_data(self, input_dir: str) -> Dict:
        """
        Load previously saved processed data.
        
        Args:
            input_dir: Directory containing saved files
            
        Returns:
            Dictionary containing loaded data
        """
        input_path = Path(input_dir)
        data_dict = {}
        
        print("\n" + "="*60)
        print("LOADING PROCESSED DATA")
        print("="*60)
        
        # Load all pickle files
        for file_path in input_path.glob("*.pkl"):
            with open(file_path, 'rb') as f:
                data_dict[file_path.stem] = pickle.load(f)
            print(f"Loaded {file_path.name}")
        
        return data_dict


def main():
    """
    Example usage of the LanguageDataProcessor.
    """
    # Initialize processor
    processor = LanguageDataProcessor(
        data_path='data/raw/language_detection.csv',
        random_state=42
    )
    
    # Load data
    # Note: Update the column names based on your actual dataset
    df = processor.load_data()
    
    # Explore data
    stats = processor.explore_data(df, text_col='Text', label_col='Language')
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.prepare_data(
        df, text_col='Text', label_col='Language'
    )
    
    # Create character-level TF-IDF features (best for language detection)
    X_train_char, X_val_char, X_test_char = processor.create_tfidf_features(
        X_train, X_val, X_test,
        analyzer='char',
        ngram_range=(2, 5),
        max_features=10000
    )
    
    # Create word-level TF-IDF features (alternative)
    X_train_word, X_val_word, X_test_word = processor.create_tfidf_features(
        X_train, X_val, X_test,
        analyzer='word',
        ngram_range=(1, 3),
        max_features=5000
    )
    
    # Save processed data
    processor.save_processed_data(
        'data/processed',
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        X_train_char=X_train_char,
        X_val_char=X_val_char,
        X_test_char=X_test_char,
        X_train_word=X_train_word,
        X_val_word=X_val_word,
        X_test_word=X_test_word
    )
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
