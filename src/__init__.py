"""
Language Detection Project - Source Package
===========================================
CS 470: Machine Learning - Fall 2025

This package contains all the source code for the language detection project.

Modules:
    - data_preprocessing: Data loading and preprocessing utilities
    - classical_ml: Classical machine learning models
    - deep_learning: Deep learning models using PyTorch
    - evaluation: Model evaluation and visualization
    - utils: Utility functions

Author: [Your Name]
"""

__version__ = '1.0.0'
__author__ = '[Your Name]'
__email__ = '[Your Email]'

from . import data_preprocessing
from . import classical_ml
from . import deep_learning
from . import evaluation
from . import utils

__all__ = [
    'data_preprocessing',
    'classical_ml',
    'deep_learning',
    'evaluation',
    'utils'
]
