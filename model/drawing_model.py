import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw
import random
import os
from pathlib import Path


class DrawingModel:
    def __init__(self, model_path='model/trained_models/best_model.h5'):
        self.model = None
        self.model_loaded = False
        self.model_path = model_path

        # Load categories
        try:
            with open('data/categories.txt', 'r') as f:
                self.categories = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            # Fallback categories
            self.categories = [
                'apple', 'banana', 'book', 'car', 'cat', 'dog', 'house',
                'tree', 'sun', 'moon', 'star', 'cloud', 'flower', 'heart'
            ]

        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                self.model_loaded = True
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print("Trained model not found. Using fallback simple model.")
                self.create_simple_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_simple_model()

    def create_simple_model(self):
        """Fallback simple model"""
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.categories), activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model_loaded = True

    def preprocess_drawing(self, drawing_data, size=28):
        """Convert drawing data to image and preprocess for model"""
        # Create a blank image
        img = Image.new('L', (256, 256), 255)
        draw = ImageDraw.Draw(img)

        # Draw the strokes
        for stroke in drawing_data:
            if len(stroke['x']) >= 2:
                for i in range(len(stroke['x']) - 1):
                    x1, y1 = stroke['x'][i], stroke['y'][i]
                    x2, y2 = stroke['x'][i + 1], stroke['y'][i + 1]
                    draw.line([x1, y1, x2, y2], fill=0, width=5)

        # Resize and normalize
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert colors
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict_drawing(self, drawing_data):
        """Predict the class of a drawing"""
        if not self.model_loaded:
            return {'class_name': 'Model not loaded', 'confidence': 0.0}

        try:
            # Preprocess the drawing
            processed_image = self.preprocess_drawing(drawing_data)

            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            # Get all predictions with confidence scores
            all_predictions = []
            for i, conf in enumerate(predictions[0]):
                all_predictions.append({
                    'class': self.categories[i],
                    'confidence': float(conf)
                })

            # Sort predictions by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

            return {
                'class_name': self.categories[predicted_class],
                'confidence': float(confidence),
                'all_predictions': all_predictions[:5]  # Top 5 predictions
            }

        except Exception as e:
            return {'class_name': f'Error: {str(e)}', 'confidence': 0.0}

    def get_random_category(self):
        """Get a random category from available classes"""
        return random.choice(self.categories)