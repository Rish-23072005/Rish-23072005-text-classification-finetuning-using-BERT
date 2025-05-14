import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import sys

def load_model():
    try:
        # Get Hugging Face token from environment variable
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("Hugging Face token not found in environment variables")
        
        # Get username from token
        from huggingface_hub import HfFolder
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

def predict(text):
    try:
        if model is None or tokenizer is None:
            return "Error: Model not loaded properly. Please check your Hugging Face token."
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get prediction and confidence
        sentiment = "Positive" if predictions[0][1] > predictions[0][0] else "Negative"
        confidence = float(predictions[0][1] if sentiment == "Positive" else predictions[0][0])
        
        return f"Sentiment: {sentiment}\nConfidence: {confidence:.2%}"
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Load model and tokenizer
model, tokenizer = load_model()

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Enter your text here..."),
    outputs=gr.Textbox(),
    title="BERT Sentiment Analysis",
    description="Enter text to analyze its sentiment using a fine-tuned BERT model.",
    examples=[
        ["This movie was absolutely fantastic! I loved every minute of it."],
        ["I really didn't enjoy this film. It was quite disappointing."],
        ["The acting was superb and the plot was engaging."],
        ["The movie was boring and the characters were poorly developed."]
    ]
)

if __name__ == "__main__":
    iface.launch() 