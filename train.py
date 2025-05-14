import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from huggingface_hub import login, HfFolder
import os
import sys

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    try:
        # Get Hugging Face token from environment variable
        token = os.getenv('HF_TOKEN')
        if not token:
            print("Error: Hugging Face token not found!")
            print("Please set your Hugging Face token as an environment variable:")
            print("export HF_TOKEN='your_token_here'")
            sys.exit(1)
        
        # Login to Hugging Face Hub
        print("Logging in to Hugging Face Hub...")
        login(token=token)
        print("Successfully logged in to Hugging Face Hub")

        # Load dataset (using IMDB as an example)
        print("Loading IMDB dataset...")
        dataset = load_dataset("imdb")
        
        # Load tokenizer and model
        print("Loading BERT model and tokenizer...")
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Tokenize the dataset
        print("Tokenizing dataset...")
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Get username from token
        username = HfFolder.get_username()
        if not username:
            raise ValueError("Could not get username from token")

        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # Reduced batch size to prevent memory issues
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,
            hub_model_id=f"{username}/bert-imdb-classifier",
            logging_steps=100,
            save_total_limit=2,
            fp16=True,  # Enable mixed precision training
        )

        # Initialize trainer
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )

        # Train the model
        print("Starting training...")
        trainer.train()

        # Push to Hub
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"Model successfully uploaded to: https://huggingface.co/{username}/bert-imdb-classifier")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 