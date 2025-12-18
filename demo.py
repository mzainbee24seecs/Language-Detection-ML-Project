"""
Language Detection - Interactive Demo
======================================
Live demonstration of the trained models

Author: Muhammad Zain
Course: CS 470 - Machine Learning
"""

import pickle
import numpy as np
import sys
from pathlib import Path

def load_models():
    """Load trained model and preprocessors."""
    try:
        with open('models/saved_models/svm_linear.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/processed/tfidf_char.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('data/processed/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, vectorizer, label_encoder
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. Please train models first.")
        print(f"Run: python main.py --train-all")
        sys.exit(1)

def clean_text(text):
    """Simple text cleaning."""
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_language(text, model, vectorizer, label_encoder, show_top_n=3):
    """Predict language of input text."""
    
    # Clean and vectorize
    cleaned_text = clean_text(text)
    if len(cleaned_text) == 0:
        print("Error: Empty text after cleaning")
        return
    
    features = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(features)[0]
    predicted_language = label_encoder.inverse_transform([prediction])[0]
    
    # Get confidence scores (decision function values)
    try:
        decision_values = model.decision_function(features)[0]
        top_n_idx = np.argsort(decision_values)[-show_top_n:][::-1]
        
        print(f"\n{'='*60}")
        print(f"Input: '{text}'")
        print(f"{'='*60}")
        print(f"\n🎯 Predicted Language: {predicted_language.upper()}")
        print(f"\n📊 Top {show_top_n} Predictions:")
        
        for i, idx in enumerate(top_n_idx, 1):
            lang = label_encoder.classes_[idx]
            score = decision_values[idx]
            bar_length = int((score + 5) * 2)  # Simple visualization
            bar = '█' * max(0, bar_length)
            print(f"  {i}. {lang:15s} {score:6.2f} {bar}")
    
    except AttributeError:
        # For models without decision_function
        print(f"\n{'='*60}")
        print(f"Input: '{text}'")
        print(f"{'='*60}")
        print(f"\n🎯 Predicted Language: {predicted_language.upper()}")

def run_demo_examples(model, vectorizer, label_encoder):
    """Run demonstration with example texts."""
    
    print("\n" + "="*60)
    print("LANGUAGE DETECTION DEMO - Example Predictions")
    print("="*60)
    
    examples = [
        ("Hello, how are you today?", "English"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Hola, ¿cómo estás?", "Spanish"),
        ("Guten Tag, wie geht es Ihnen?", "German"),
        ("Ciao, come stai?", "Italian"),
        ("Olá, como você está?", "Portuguese"),
        ("Hallo, hoe gaat het met je?", "Dutch"),
        ("Привет, как дела?", "Russian"),
        ("こんにちは、元気ですか？", "Japanese"),
        ("你好，你好吗？", "Chinese"),
    ]
    
    correct = 0
    for text, expected in examples:
        cleaned_text = clean_text(text)
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)[0]
        predicted_lang = label_encoder.inverse_transform([prediction])[0]
        
        status = "✓" if predicted_lang.lower() == expected.lower() else "✗"
        print(f"\n{status} {expected:12s} | {text:40s} → {predicted_lang}")
        
        if predicted_lang.lower() == expected.lower():
            correct += 1
    
    accuracy = (correct / len(examples)) * 100
    print(f"\n{'='*60}")
    print(f"Demo Accuracy: {correct}/{len(examples)} ({accuracy:.1f}%)")
    print(f"{'='*60}")

def interactive_mode(model, vectorizer, label_encoder):
    """Interactive prediction mode."""
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nEnter text in any language to detect it.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'examples' to see demo examples again")
    print("  - Type 'help' for language list")
    print()
    
    while True:
        try:
            user_input = input("\n📝 Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for trying the language detector!")
                break
            
            if user_input.lower() == 'examples':
                run_demo_examples(model, vectorizer, label_encoder)
                continue
            
            if user_input.lower() == 'help':
                print("\nSupported Languages:")
                languages = sorted(label_encoder.classes_)
                for i, lang in enumerate(languages, 1):
                    print(f"  {i:2d}. {lang}")
                continue
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            predict_language(user_input, model, vectorizer, label_encoder)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def main():
    """Main demo function."""
    
    print("\n" + "="*60)
    print("LANGUAGE DETECTION PROJECT DEMO")
    print("="*60)
    print("\nCS 470 - Machine Learning | Fall 2025")
    print("Student: Zain")
    print("Model: Linear SVM (98.58% accuracy)")
    print("="*60)
    
    # Load models
    print("\n⏳ Loading models...")
    model, vectorizer, label_encoder = load_models()
    print("✓ Models loaded successfully!")
    
    # Show menu
    while True:
        print("\n" + "="*60)
        print("DEMO OPTIONS")
        print("="*60)
        print("1. Run demo examples")
        print("2. Interactive mode (enter your own text)")
        print("3. Show project statistics")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            run_demo_examples(model, vectorizer, label_encoder)
        elif choice == '2':
            interactive_mode(model, vectorizer, label_encoder)
        elif choice == '3':
            print("\n" + "="*60)
            print("PROJECT STATISTICS")
            print("="*60)
            print("\nDataset:")
            print("  • Total samples: 10,337")
            print("  • Languages: 17")
            print("  • Split: 70% train, 15% val, 15% test")
            print("\nModel Performance (Test Set):")
            print("  • Linear SVM:         98.58% accuracy ⭐")
            print("  • Logistic Regression: 98.52% accuracy")
            print("  • CNN (Deep Learning): 98.00% accuracy")
            print("  • Naive Bayes:        97.29% accuracy")
            print("  • Random Forest:      97.03% accuracy")
            print("\nKey Findings:")
            print("  • Character n-grams (2-5) most effective")
            print("  • Classical ML outperformed Deep Learning")
            print("  • Training time: 20-40 minutes total")
            print("="*60)
        elif choice == '4':
            print("\n👋 Thank you for viewing the demo!")
            break
        else:
            print("Invalid option. Please select 1-4.")

if __name__ == "__main__":
    main()

