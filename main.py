"""
Main Pipeline Script for Language Detection Project
====================================================
This script orchestrates the entire ML pipeline:
1. Data preprocessing
2. Classical ML model training
3. Deep learning model training
4. Model evaluation and comparison
5. Results visualization

Author: Muhammad Zain
Course: CS 470 - Machine Learning
Project: Language Detection (Multi-class Classification)
"""

import argparse
import sys
import pickle
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_preprocessing import LanguageDataProcessor
from src.classical_ml import ClassicalMLModels
from src.deep_learning import (
    set_seed, build_vocab, TextDataset, CharCNN, BiLSTM, 
    HybridCNNLSTM, DeepLearningTrainer
)
from src.evaluation import ModelEvaluator
import torch
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Language Detection - ML Pipeline'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/language_detection.csv',
        help='Path to the raw dataset CSV file'
    )
    
    parser.add_argument(
        '--text-col',
        type=str,
        default='Text',
        help='Name of the text column in the dataset'
    )
    
    parser.add_argument(
        '--label-col',
        type=str,
        default='Language',
        help='Name of the label column in the dataset'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run data preprocessing'
    )
    
    parser.add_argument(
        '--train-classical',
        action='store_true',
        help='Train classical ML models'
    )
    
    parser.add_argument(
        '--train-dl',
        action='store_true',
        help='Train deep learning models'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate all trained models'
    )
    
    parser.add_argument(
        '--train-all',
        action='store_true',
        help='Run the entire pipeline (preprocess + train + evaluate)'
    )
    
    parser.add_argument(
        '--dl-model',
        type=str,
        default='cnn',
        choices=['cnn', 'bilstm', 'hybrid'],
        help='Deep learning model architecture to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs for deep learning'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for deep learning'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def run_preprocessing(args):
    """
    Run data preprocessing pipeline.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    # Initialize processor
    processor = LanguageDataProcessor(
        data_path=args.data_path,
        random_state=args.seed
    )
    
    # Load data
    df = processor.load_data()
    
    # Explore data
    stats = processor.explore_data(df, text_col=args.text_col, label_col=args.label_col)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.prepare_data(
        df, text_col=args.text_col, label_col=args.label_col
    )
    
    # Create character-level TF-IDF features (best for language detection)
    print("\nCreating character-level TF-IDF features...")
    X_train_char, X_val_char, X_test_char = processor.create_tfidf_features(
        X_train, X_val, X_test,
        analyzer='char',
        ngram_range=(2, 5),
        max_features=10000
    )
    
    # Create word-level TF-IDF features (alternative)
    print("\nCreating word-level TF-IDF features...")
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
    
    print("\n✓ Data preprocessing completed successfully!")


def run_classical_ml(args):
    """
    Train classical ML models.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("STEP 2: CLASSICAL MACHINE LEARNING")
    print("="*70)
    
    # Load processed data
    print("\nLoading preprocessed data...")
    with open('data/processed/X_train_char.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/processed/X_val_char.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open('data/processed/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/processed/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    
    # Initialize model manager
    ml_models = ClassicalMLModels(random_state=args.seed)
    
    # Store results
    results = {}
    
    print("\nTraining classical ML models...")
    print("(This may take several minutes depending on your hardware)")
    
    # Train Linear SVM
    print("\n[1/4] Training Linear SVM...")
    _, results['Linear SVM'] = ml_models.train_svm_linear(
        X_train, y_train, X_val, y_val, tune_hyperparams=True
    )
    ml_models.save_model('svm_linear', 'models/saved_models')
    
    # Train Random Forest
    print("\n[2/4] Training Random Forest...")
    _, results['Random Forest'] = ml_models.train_random_forest(
        X_train, y_train, X_val, y_val, tune_hyperparams=True
    )
    ml_models.save_model('random_forest', 'models/saved_models')
    
    # Train Logistic Regression
    print("\n[3/4] Training Logistic Regression...")
    _, results['Logistic Regression'] = ml_models.train_logistic_regression(
        X_train, y_train, X_val, y_val, tune_hyperparams=True
    )
    ml_models.save_model('logistic_regression', 'models/saved_models')
    
    # Train Naive Bayes
    print("\n[4/4] Training Naive Bayes...")
    _, results['Naive Bayes'] = ml_models.train_naive_bayes(
        X_train, y_train, X_val, y_val
    )
    ml_models.save_model('naive_bayes', 'models/saved_models')
    
    # Compare models
    print("\n" + "="*60)
    print("CLASSICAL ML MODEL COMPARISON")
    print("="*60)
    comparison_df = ml_models.compare_models(results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv('results/metrics/classical_ml_comparison.csv', index=False)
    
    print("\n✓ Classical ML training completed successfully!")


def run_deep_learning(args):
    """
    Train deep learning models.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("STEP 3: DEEP LEARNING")
    print("="*70)
    
    set_seed(args.seed)
    
    # Load text data
    print("\nLoading preprocessed data...")
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
    
    # Save vocabulary
    Path('models/saved_models').mkdir(parents=True, exist_ok=True)
    with open('models/saved_models/char_to_idx.pkl', 'wb') as f:
        pickle.dump(char_to_idx, f)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TextDataset(X_train, y_train, char_to_idx, max_length=200)
    val_dataset = TextDataset(X_val, y_val, char_to_idx, max_length=200)
    test_dataset = TextDataset(X_test, y_test, char_to_idx, max_length=200)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Select model architecture
    print(f"\nInitializing {args.dl_model.upper()} model...")
    if args.dl_model == 'cnn':
        model = CharCNN(
            vocab_size=len(char_to_idx),
            embedding_dim=256,
            num_classes=num_classes,
            num_filters=128,
            kernel_sizes=[3, 4, 5],
            dropout=0.5
        )
    elif args.dl_model == 'bilstm':
        model = BiLSTM(
            vocab_size=len(char_to_idx),
            embedding_dim=256,
            hidden_dim=128,
            num_classes=num_classes,
            num_layers=2,
            dropout=0.3
        )
    else:  # hybrid
        model = HybridCNNLSTM(
            vocab_size=len(char_to_idx),
            embedding_dim=256,
            num_filters=128,
            kernel_size=3,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.5
        )
    
    # Train model
    trainer = DeepLearningTrainer(model)
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=args.epochs, learning_rate=0.001, patience=5
    )
    
    # Plot training history
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    trainer.plot_training_history(f'results/figures/training_history_{args.dl_model}.png')
    
    # Save model
    torch.save(model.state_dict(), f'models/saved_models/{args.dl_model}_model.pth')
    
    print(f"\n✓ Deep learning ({args.dl_model.upper()}) training completed successfully!")


def run_evaluation(args):
    """
    Evaluate all trained models.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION & COMPARISON")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    with open('data/processed/X_test_char.pkl', 'rb') as f:
        X_test_tfidf = pickle.load(f)
    with open('data/processed/X_test.pkl', 'rb') as f:
        X_test_text = pickle.load(f)
    with open('data/processed/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open('data/processed/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    class_names = label_encoder.classes_
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names=class_names)
    
    # Evaluate classical ML models
    print("\n" + "-"*60)
    print("EVALUATING CLASSICAL ML MODELS")
    print("-"*60)
    
    classical_models = ['svm_linear', 'random_forest', 'logistic_regression', 'naive_bayes']
    classical_predictions = {}
    
    for model_name in classical_models:
        model_path = f'models/saved_models/{model_name}.pkl'
        if Path(model_path).exists():
            print(f"\nEvaluating {model_name}...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            y_pred = model.predict(X_test_tfidf)
            classical_predictions[model_name] = y_pred
            
            # Evaluate
            evaluator.evaluate_model(y_test, y_pred, model_name)
            
            # Plot confusion matrix
            evaluator.plot_confusion_matrix(
                y_test, y_pred, model_name,
                save_path=f'results/figures/cm_{model_name}.png'
            )
    
    # Evaluate deep learning model
    print("\n" + "-"*60)
    print("EVALUATING DEEP LEARNING MODEL")
    print("-"*60)
    
    if Path('models/saved_models/char_to_idx.pkl').exists():
        with open('models/saved_models/char_to_idx.pkl', 'rb') as f:
            char_to_idx = pickle.load(f)
        
        # Find the trained DL model
        for dl_type in ['cnn', 'bilstm', 'hybrid']:
            model_path = f'models/saved_models/{dl_type}_model.pth'
            if Path(model_path).exists():
                print(f"\nEvaluating {dl_type.upper()} model...")
                
                # Create dataset and loader
                test_dataset = TextDataset(X_test_text, y_test, char_to_idx, max_length=200)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                
                # Load model
                num_classes = len(class_names)
                if dl_type == 'cnn':
                    model = CharCNN(len(char_to_idx), 256, num_classes)
                elif dl_type == 'bilstm':
                    model = BiLSTM(len(char_to_idx), 256, 128, num_classes)
                else:
                    model = HybridCNNLSTM(len(char_to_idx), 256, 128, 3, 128, num_classes)
                
                model.load_state_dict(torch.load(model_path))
                trainer = DeepLearningTrainer(model)
                
                # Predict
                y_pred_dl, y_pred_proba = trainer.predict(test_loader)
                
                # Evaluate
                evaluator.evaluate_model(y_test, y_pred_dl, f'{dl_type.upper()}')
                
                # Plot confusion matrix
                evaluator.plot_confusion_matrix(
                    y_test, y_pred_dl, f'{dl_type.upper()}',
                    save_path=f'results/figures/cm_{dl_type}.png'
                )
                
                # Plot ROC curves
                evaluator.plot_roc_curves(
                    y_test, y_pred_proba, f'{dl_type.upper()}',
                    save_path=f'results/figures/roc_{dl_type}.png'
                )
                
                break
    
    # Compare all models
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    comparison_df = evaluator.compare_models('results/metrics/final_comparison.csv')
    
    # Plot comparison
    evaluator.plot_model_comparison(
        metrics=['accuracy', 'f1_macro'],
        save_path='results/figures/model_comparison.png'
    )
    
    # Statistical significance testing
    if len(classical_predictions) >= 2:
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        models = list(classical_predictions.keys())
        evaluator.mcnemar_test(
            y_test,
            classical_predictions[models[0]],
            classical_predictions[models[1]],
            models[0],
            models[1]
        )
    
    print("\n✓ Evaluation completed successfully!")
    print(f"\nResults saved to:")
    print(f"  - Metrics: results/metrics/")
    print(f"  - Figures: results/figures/")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("LANGUAGE DETECTION PROJECT - ML PIPELINE")
    print("CS 470: Machine Learning - Fall 2025")
    print("="*70)
    print(f"\nRandom Seed: {args.seed}")
    
    try:
        if args.train_all:
            # Run entire pipeline
            run_preprocessing(args)
            run_classical_ml(args)
            run_deep_learning(args)
            run_evaluation(args)
        else:
            # Run individual components
            if args.preprocess:
                run_preprocessing(args)
            
            if args.train_classical:
                run_classical_ml(args)
            
            if args.train_dl:
                run_deep_learning(args)
            
            if args.evaluate:
                run_evaluation(args)
            
            if not (args.preprocess or args.train_classical or args.train_dl or args.evaluate):
                print("\nNo action specified. Use --help to see available options.")
                print("\nQuick start:")
                print("  python main.py --train-all    # Run entire pipeline")
                print("  python main.py --preprocess   # Preprocess data only")
                print("  python main.py --evaluate     # Evaluate trained models")
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
