"""
Utility Functions for Language Detection Project
=================================================
Common utility functions used across the project.

Course: CS 470 - Machine Learning
Project: Language Detection (Multi-class Classification)
"""

import numpy as np
import random
import torch
import os
from pathlib import Path


def set_all_seeds(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def count_parameters(model):
    """
    Count trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_section_header(title: str, width: int = 70):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the header
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def save_results_summary(results: dict, filepath: str):
    """
    Save results summary to a text file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save the file
    """
    ensure_dir(Path(filepath).parent)
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Language Detection Project - Results Summary\n")
        f.write("="*70 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results summary saved to {filepath}")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")

