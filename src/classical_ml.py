"""
Classical Machine Learning Models for Language Detection
=========================================================
This module implements classical ML algorithms including:
- Support Vector Machine (SVM)
- Random Forest Classifier
- Logistic Regression
- Naive Bayes
- XGBoost

Includes hyperparameter tuning using GridSearchCV and cross-validation.

Author: Muhammad Zain
Course: CS 470 - Machine Learning
Project: Language Detection (Multi-class Classification)
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class ClassicalMLModels:
    """
    Class for training and evaluating classical ML models.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the classical ML models manager.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 means use all processors)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.best_params = {}
        self.training_times = {}
        
    def train_svm_linear(self, X_train, y_train, X_val, y_val,
                        tune_hyperparams: bool = True) -> Tuple[Any, Dict]:
        """
        Train Linear SVM with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Tuple of (trained model, results dictionary)
        """
        print("\n" + "="*60)
        print("TRAINING LINEAR SVM")
        print("="*60)
        
        start_time = time.time()
        
        if tune_hyperparams:
            print("Performing Grid Search for hyperparameter tuning...")
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'max_iter': [1000]
            }
            
            svm = LinearSVC(random_state=self.random_state, dual=True)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy',
                n_jobs=self.n_jobs, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['svm_linear'] = grid_search.best_params_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            model = LinearSVC(C=1.0, random_state=self.random_state, max_iter=1000)
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['svm_linear'] = training_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, average='macro'),
            'val_recall': recall_score(y_val, y_val_pred, average='macro'),
            'val_f1': f1_score(y_val, y_val_pred, average='macro'),
            'training_time': training_time
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation F1-Score (Macro): {results['val_f1']:.4f}")
        
        self.models['svm_linear'] = model
        return model, results
    
    def train_svm_rbf(self, X_train, y_train, X_val, y_val,
                      tune_hyperparams: bool = True) -> Tuple[Any, Dict]:
        """
        Train SVM with RBF kernel and optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Tuple of (trained model, results dictionary)
        """
        print("\n" + "="*60)
        print("TRAINING SVM WITH RBF KERNEL")
        print("="*60)
        
        start_time = time.time()
        
        if tune_hyperparams:
            print("Performing Grid Search for hyperparameter tuning...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
            
            svm = SVC(kernel='rbf', random_state=self.random_state)
            grid_search = GridSearchCV(
                svm, param_grid, cv=3, scoring='accuracy',
                n_jobs=self.n_jobs, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['svm_rbf'] = grid_search.best_params_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state)
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['svm_rbf'] = training_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, average='macro'),
            'val_recall': recall_score(y_val, y_val_pred, average='macro'),
            'val_f1': f1_score(y_val, y_val_pred, average='macro'),
            'training_time': training_time
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation F1-Score (Macro): {results['val_f1']:.4f}")
        
        self.models['svm_rbf'] = model
        return model, results
    
    def train_random_forest(self, X_train, y_train, X_val, y_val,
                           tune_hyperparams: bool = True) -> Tuple[Any, Dict]:
        """
        Train Random Forest with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Tuple of (trained model, results dictionary)
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        start_time = time.time()
        
        if tune_hyperparams:
            print("Performing Grid Search for hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='accuracy',
                n_jobs=self.n_jobs, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            model = RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['random_forest'] = training_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, average='macro'),
            'val_recall': recall_score(y_val, y_val_pred, average='macro'),
            'val_f1': f1_score(y_val, y_val_pred, average='macro'),
            'training_time': training_time
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation F1-Score (Macro): {results['val_f1']:.4f}")
        
        self.models['random_forest'] = model
        return model, results
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val,
                                  tune_hyperparams: bool = True) -> Tuple[Any, Dict]:
        """
        Train Logistic Regression with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Tuple of (trained model, results dictionary)
        """
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        start_time = time.time()
        
        if tune_hyperparams:
            print("Performing Grid Search for hyperparameter tuning...")
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [1000]
            }
            
            lr = LogisticRegression(random_state=self.random_state, n_jobs=self.n_jobs)
            grid_search = GridSearchCV(
                lr, param_grid, cv=5, scoring='accuracy',
                n_jobs=self.n_jobs, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['logistic_regression'] = grid_search.best_params_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state, n_jobs=self.n_jobs
            )
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['logistic_regression'] = training_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, average='macro'),
            'val_recall': recall_score(y_val, y_val_pred, average='macro'),
            'val_f1': f1_score(y_val, y_val_pred, average='macro'),
            'training_time': training_time
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation F1-Score (Macro): {results['val_f1']:.4f}")
        
        self.models['logistic_regression'] = model
        return model, results
    
    def train_naive_bayes(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        """
        Train Multinomial Naive Bayes (best for text classification).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (trained model, results dictionary)
        """
        print("\n" + "="*60)
        print("TRAINING MULTINOMIAL NAIVE BAYES")
        print("="*60)
        
        start_time = time.time()
        
        print("Training Naive Bayes...")
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['naive_bayes'] = training_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, average='macro'),
            'val_recall': recall_score(y_val, y_val_pred, average='macro'),
            'val_f1': f1_score(y_val, y_val_pred, average='macro'),
            'training_time': training_time
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation F1-Score (Macro): {results['val_f1']:.4f}")
        
        self.models['naive_bayes'] = model
        return model, results
    
    def save_model(self, model_name: str, output_dir: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            output_dir: Directory to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / f"{model_name}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        print(f"Saved {model_name} to {file_path}")
    
    def load_model(self, model_name: str, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            model_path: Path to the saved model file
        """
        with open(model_path, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        
        print(f"Loaded {model_name} from {model_path}")
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create a comparison table of model performances.
        
        Args:
            results_dict: Dictionary of model results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Train Accuracy': f"{results['train_accuracy']:.4f}",
                'Val Accuracy': f"{results['val_accuracy']:.4f}",
                'Val Precision': f"{results['val_precision']:.4f}",
                'Val Recall': f"{results['val_recall']:.4f}",
                'Val F1-Score': f"{results['val_f1']:.4f}",
                'Training Time (s)': f"{results['training_time']:.2f}"
            })
        
        df = pd.DataFrame(comparison_data)
        return df


def main():
    """
    Example usage of ClassicalMLModels.
    """
    # Load processed data
    print("Loading processed data...")
    with open('data/processed/X_train_char.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/processed/X_val_char.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open('data/processed/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/processed/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    
    # Initialize model manager
    ml_models = ClassicalMLModels(random_state=42)
    
    # Store results
    results = {}
    
    # Train Linear SVM
    _, results['Linear SVM'] = ml_models.train_svm_linear(
        X_train, y_train, X_val, y_val, tune_hyperparams=True
    )
    ml_models.save_model('svm_linear', 'models/saved_models')
    
    # Train Random Forest
    _, results['Random Forest'] = ml_models.train_random_forest(
        X_train, y_train, X_val, y_val, tune_hyperparams=True
    )
    ml_models.save_model('random_forest', 'models/saved_models')
    
    # Train Logistic Regression
    _, results['Logistic Regression'] = ml_models.train_logistic_regression(
        X_train, y_train, X_val, y_val, tune_hyperparams=True
    )
    ml_models.save_model('logistic_regression', 'models/saved_models')
    
    # Train Naive Bayes
    _, results['Naive Bayes'] = ml_models.train_naive_bayes(
        X_train, y_train, X_val, y_val
    )
    ml_models.save_model('naive_bayes', 'models/saved_models')
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison_df = ml_models.compare_models(results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('results/metrics/classical_ml_comparison.csv', index=False)
    print("\nComparison saved to results/metrics/classical_ml_comparison.csv")


if __name__ == "__main__":
    main()

