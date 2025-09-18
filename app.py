from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import tensorflow as tf
from model.drawing_model import DrawingModel
import json

app = Flask(__name__)

# Initialize the drawing model
drawing_model = DrawingModel()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get drawing data from request
        data = request.json
        drawing_data = data.get('drawing')

        if not drawing_data:
            return jsonify({'error': 'No drawing data provided'}), 400

        # Convert drawing to image and predict
        prediction = drawing_model.predict_drawing(drawing_data)

        return jsonify({
            'prediction': prediction['class_name'],
            'confidence': float(prediction['confidence']),
            'all_predictions': prediction['all_predictions']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/random_word', methods=['GET'])
def get_random_word():
    """Get a random word from the Quick Draw dataset categories"""
    try:
        word = drawing_model.get_random_category()
        return jsonify({'word': word})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': drawing_model.model_loaded})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)