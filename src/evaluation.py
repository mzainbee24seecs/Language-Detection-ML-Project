"""
Evaluation and Visualization Module
====================================
Comprehensive evaluation metrics and visualization for both classical ML 
and deep learning models. Includes confusion matrix, classification report,
ROC curves, and statistical significance testing.

Author: [Your Name]
Course: CS 470 - Machine Learning
Project: Language Detection (Multi-class Classification)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.results = {}
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str) -> Dict:
        """
        Evaluate a model with multiple metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
        
        # Print results
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Precision (Macro):  {precision_macro:.4f}")
        print(f"Precision (Weighted): {precision_weighted:.4f}")
        print(f"Recall (Macro):     {recall_macro:.4f}")
        print(f"Recall (Weighted):  {recall_weighted:.4f}")
        print(f"F1-Score (Macro):   {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str, save_path: str = None,
                             normalize: bool = True):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the figure
            normalize: Whether to normalize the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names if self.class_names is not None else 'auto',
                   yticklabels=self.class_names if self.class_names is not None else 'auto')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, save_path: str = None):
        """
        Generate and plot classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the figure
        """
        # Generate classification report as dictionary
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names if self.class_names is not None else None,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame for visualization
        df_report = pd.DataFrame(report).transpose()
        
        # Print text version
        print(f"\n{model_name} - Classification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.class_names if self.class_names is not None else None,
            zero_division=0
        ))
        
        # Plot heatmap (excluding support row and summary rows)
        plot_data = df_report.iloc[:-3, :-1]  # Exclude accuracy, macro avg, weighted avg, and support
        
        plt.figure(figsize=(10, len(plot_data) * 0.5 + 2))
        sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
        plt.title(f'Classification Report - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Class', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification report saved to {save_path}")
        
        plt.show()
        
        return df_report
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str, save_path: str = None):
        """
        Plot ROC curves for multi-class classification (one-vs-rest).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for each class
            model_name: Name of the model
            save_path: Path to save the figure
        """
        n_classes = len(np.unique(y_true))
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            class_name = self.class_names[i] if self.class_names is not None else f'Class {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves (One-vs-Rest) - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, save_path: str = None) -> pd.DataFrame:
        """
        Create a comparison table of all evaluated models.
        
        Args:
            save_path: Path to save the comparison table
            
        Returns:
            DataFrame with model comparison
        """
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}\n")
        
        if not self.results:
            print("No models have been evaluated yet.")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        
        # Sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
        
        print(df.to_string())
        
        if save_path:
            df.to_csv(save_path)
            print(f"\nComparison table saved to {save_path}")
        
        return df
    
    def plot_model_comparison(self, metrics: List[str] = ['accuracy', 'f1_macro'],
                             save_path: str = None):
        """
        Plot comparison of models across different metrics.
        
        Args:
            metrics: List of metrics to compare
            save_path: Path to save the figure
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return
        
        df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            data = df[metric].sort_values(ascending=False)
            
            axes[idx].barh(range(len(data)), data.values, color='steelblue')
            axes[idx].set_yticks(range(len(data)))
            axes[idx].set_yticklabels(data.index)
            axes[idx].set_xlabel('Score', fontsize=12)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', 
                              fontsize=14, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(data.values):
                axes[idx].text(v + 0.01, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def mcnemar_test(self, y_true: np.ndarray, y_pred1: np.ndarray, 
                    y_pred2: np.ndarray, model1_name: str, 
                    model2_name: str) -> Dict:
        """
        Perform McNemar's test for statistical significance between two models.
        
        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print(f"McNEMAR'S TEST: {model1_name} vs {model2_name}")
        print(f"{'='*60}")
        
        # Create contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        both_correct = np.sum(correct1 & correct2)
        both_incorrect = np.sum(~correct1 & ~correct2)
        model1_only = np.sum(correct1 & ~correct2)
        model2_only = np.sum(~correct1 & correct2)
        
        # Contingency table
        contingency = np.array([[both_correct, model2_only],
                               [model1_only, both_incorrect]])
        
        print(f"\nContingency Table:")
        print(f"                    Model 2 Correct | Model 2 Incorrect")
        print(f"Model 1 Correct     {both_correct:8d}       | {model1_only:8d}")
        print(f"Model 1 Incorrect   {model2_only:8d}       | {both_incorrect:8d}")
        
        # Perform McNemar's test
        # Using continuity correction
        statistic = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        print(f"\nTest Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"Conclusion: The difference is statistically significant (p < 0.05)")
        else:
            print(f"Conclusion: The difference is NOT statistically significant (p >= 0.05)")
        
        return {
            'model1': model1_name,
            'model2': model2_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'contingency_table': contingency
        }
    
    def per_class_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str, save_path: str = None):
        """
        Analyze per-class performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the figure
        """
        # Calculate per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Create DataFrame
        class_names = self.class_names if self.class_names is not None else [f'Class {i}' for i in range(len(precision))]
        df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"\n{model_name} - Per-Class Performance:")
        print(df.to_string(index=False))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.4)))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax.barh(x - width, precision, width, label='Precision', color='steelblue')
        ax.barh(x, recall, width, label='Recall', color='darkorange')
        ax.barh(x + width, f1, width, label='F1-Score', color='green')
        
        ax.set_yticks(x)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Performance - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class analysis saved to {save_path}")
        
        plt.show()
        
        return df


def main():
    """
    Example usage of ModelEvaluator.
    """
    # Load test data and predictions
    print("Loading test data and predictions...")
    
    # This is a placeholder - you would load actual predictions
    # For demonstration, we'll show how to use the evaluator
    
    # Example: Load label encoder to get class names
    with open('data/processed/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    class_names = label_encoder.classes_
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names=class_names)
    
    # Example evaluation workflow:
    # 1. Load true labels
    # with open('data/processed/y_test.pkl', 'rb') as f:
    #     y_test = pickle.load(f)
    
    # 2. Load predictions from each model
    # y_pred_svm = ...
    # y_pred_rf = ...
    # y_pred_dl = ...
    
    # 3. Evaluate each model
    # evaluator.evaluate_model(y_test, y_pred_svm, 'Linear SVM')
    # evaluator.plot_confusion_matrix(y_test, y_pred_svm, 'Linear SVM', 
    #                                'results/figures/cm_svm.png')
    
    # 4. Compare models
    # evaluator.compare_models('results/metrics/model_comparison.csv')
    # evaluator.plot_model_comparison(save_path='results/figures/comparison.png')
    
    # 5. Statistical testing
    # evaluator.mcnemar_test(y_test, y_pred_svm, y_pred_rf, 'SVM', 'Random Forest')
    
    print("\nEvaluation module loaded successfully!")
    print("Use the ModelEvaluator class to evaluate your trained models.")


if __name__ == "__main__":
    main()
