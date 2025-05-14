#Text-classification-finetuning-using-BERT

This project demonstrates how to fine-tune a BERT model for sentiment analysis using the Hugging Face Transformers library and provides both a Flask API and a Gradio interface for making predictions.

## Features

- Fine-tuned BERT model for sentiment analysis
- Flask API for production deployment
- Gradio interface for easy testing and demonstration
- Support for both positive and negative sentiment classification
- Confidence scores for predictions

## Project Structure

```
.
├── train.py              # Script for fine-tuning BERT
├── app.py               # Flask API implementation
├── gradio_app.py        # Gradio interface
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Rish-23072005/Rish-23072005-text-classification-finetuning-using-BERT.git
cd Rish-23072005-text-classification-finetuning-using-BERT
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with write access
   - Update the token in the code or set it as an environment variable

## Usage

### Training the Model

To fine-tune the BERT model:
```bash
python train.py
```

### Using the Flask API

To start the Flask API:
```bash
python app.py
```

The API will be available at `http://localhost:5000` with the following endpoints:
- POST `/predict`: Send text for sentiment analysis
- GET `/health`: Check API health status

Example API request:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic!"}'
```

### Using the Gradio Interface

To start the Gradio interface:
```bash
python gradio_app.py
```

This will launch a web interface where you can:
- Enter text for sentiment analysis
- See the prediction and confidence score
- Try example inputs

## Model

The fine-tuned model will be uploaded to Hugging Face Hub and can be accessed using the model ID provided after training.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- BERT model by Google
- IMDB dataset for training 
