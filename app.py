from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from huggingface_hub import HfFolder

app = Flask(__name__)

def load_model():
    try:
        # Get Hugging Face token from environment variable
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("Hugging Face token not found in environment variables")
        
        # Get username from token
        username = HfFolder.get_username()
        if not username:
            raise ValueError("Could not get username from token")
        
        model_name = f"{username}/bert-imdb-classifier"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# Load model and tokenizer
model, tokenizer = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or tokenizer is None:
            return jsonify({
                'error': 'Model not loaded properly. Please check your Hugging Face token.'
            }), 500

        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided in request'
            }), 400

        text = data['text']

        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get prediction and confidence
        sentiment = "Positive" if predictions[0][1] > predictions[0][0] else "Negative"
        confidence = float(predictions[0][1] if sentiment == "Positive" else predictions[0][0])

        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    if model is None or tokenizer is None:
        print("Error: Model or tokenizer not loaded. Please check your Hugging Face token.")
    else:
        print("Model loaded successfully!")
        app.run(debug=True) 