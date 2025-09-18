import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class QuickDrawPreprocessor:
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)

        # Load categories
        with open('data/categories.txt', 'r') as f:
            self.categories = [line.strip() for line in f.readlines()]

    def load_category_data(self, category):
        """Load data for a specific category"""
        file_path = os.path.join(self.raw_data_dir, f"{category}.npy")
        if os.path.exists(file_path):
            return np.load(file_path)
        return None

    def preprocess_images(self, images, target_size=(28, 28)):
        """Preprocess images for training"""
        # Reshape and normalize
        images = images.reshape(-1, 28, 28, 1)
        images = images.astype('float32') / 255.0

        # Add data augmentation (simple version)
        # You can use tf.keras.preprocessing.image.ImageDataGenerator for more augmentation
        return images

    def create_dataset(self, samples_per_category=5000, test_size=0.2):
        """Create training and testing datasets"""
        X, y = [], []

        print("Loading and preprocessing data...")

        for label, category in enumerate(tqdm(self.categories)):
            data = self.load_category_data(category)
            if data is not None:
                # Take specified number of samples
                n_samples = min(samples_per_category, len(data))
                category_data = data[:n_samples]

                # Preprocess
                processed_data = self.preprocess_images(category_data)

                X.extend(processed_data)
                y.extend([label] * n_samples)

        X = np.array(X)
        y = np.array(y)

        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Save processed data
        np.save(os.path.join(self.processed_data_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_data_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(self.processed_data_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_data_dir, 'y_test.npy'), y_test)

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def visualize_samples(self, num_samples=5):
        """Visualize sample images from each category"""
        fig, axes = plt.subplots(len(self.categories), num_samples,
                                 figsize=(15, 2 * len(self.categories)))

        if len(self.categories) == 1:
            axes = axes.reshape(1, -1)

        for i, category in enumerate(self.categories):
            data = self.load_category_data(category)
            if data is not None:
                samples = data[:num_samples]
                for j in range(num_samples):
                    if j < len(samples):
                        img = samples[j].reshape(28, 28)
                        axes[i, j].imshow(img, cmap='gray')
                        axes[i, j].axis('off')
                        if j == 0:
                            axes[i, j].set_title(category, fontsize=10)

        plt.tight_layout()
        plt.savefig('data/sample_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    preprocessor = QuickDrawPreprocessor()
    preprocessor.visualize_samples(num_samples=5)
    X_train, X_test, y_train, y_test = preprocessor.create_dataset(samples_per_category=5000)