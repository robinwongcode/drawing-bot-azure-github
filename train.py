#!/usr/bin/env python3
"""
Complete training pipeline for Quick Draw model
"""

import sys
import os

# Add scripts to path
sys.path.append('scripts')

from download_data import QuickDrawDownloader
from preprocess_data import QuickDrawPreprocessor
from train_model import QuickDrawTrainer


def main():
    print("🚀 Starting Quick Draw Model Training Pipeline")
    print("=" * 50)

    # Step 1: Download data
    print("\n1. Downloading data...")
    downloader = QuickDrawDownloader()
    downloader.download_all_categories(max_samples_per_category=10000)

    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = QuickDrawPreprocessor()
    preprocessor.visualize_samples(num_samples=5)
    preprocessor.create_dataset(samples_per_category=5000)

    # Step 3: Train model
    print("\n3. Training model...")
    trainer = QuickDrawTrainer()
    model, history = trainer.train_model(epochs=50)

    # Step 4: Evaluate
    print("\n4. Evaluating model...")
    test_acc, y_pred = trainer.evaluate_model(model)
    trainer.plot_training_history(history)

    # Load test data for confusion matrix
    _, X_test, _, y_test = trainer.load_data()
    trainer.plot_confusion_matrix(y_test, y_pred)

    print(f"\n✅ Training complete! Model saved to model/trained_models/")
    print(f"📊 Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()