import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class QuickDrawTrainer:
    def __init__(self, processed_data_dir='data/processed', model_dir='model/trained_models'):
        self.processed_data_dir = processed_data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Load categories
        with open('data/categories.txt', 'r') as f:
            self.categories = [line.strip() for line in f.readlines()]

        self.num_classes = len(self.categories)

    def load_data(self):
        """Load preprocessed data"""
        X_train = np.load(os.path.join(self.processed_data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(self.processed_data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(self.processed_data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(self.processed_data_dir, 'y_test.npy'))

        return X_train, X_test, y_train, y_test

    def create_advanced_model(self, input_shape=(28, 28, 1)):
        """Create a more advanced CNN model"""
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu',
                          input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def train_model(self, epochs=50, batch_size=128):
        """Train the model"""
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()

        # Create model
        model = self.create_advanced_model()

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]

        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        model.save(os.path.join(self.model_dir, 'final_model.h5'))

        return model, history

    def evaluate_model(self, model):
        """Evaluate the trained model"""
        X_train, X_test, y_train, y_test = self.load_data()

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Testing Loss: {test_loss:.4f}")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes,
                                    target_names=self.categories))

        return test_acc, y_pred_classes

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.categories,
                    yticklabels=self.categories)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    trainer = QuickDrawTrainer()

    # Train model
    model, history = trainer.train_model(epochs=50)

    # Evaluate
    test_acc, y_pred = trainer.evaluate_model(model)

    # Plot results
    trainer.plot_training_history(history)

    # Load test data for confusion matrix
    _, X_test, _, y_test = trainer.load_data()
    y_true = y_test
    trainer.plot_confusion_matrix(y_true, y_pred)

    print(f"Model training complete! Test accuracy: {test_acc:.4f}")