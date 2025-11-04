import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

MODEL_NAME = os.getenv("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment")
DATA_PATH = "data/train.csv"
OUTPUT_DIR = "models/finetuned"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    return Dataset.from_pandas(df)

def tokenize(batch):
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

def train_model():
    dataset = load_data()
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Fine-tuned model saved at:", OUTPUT_DIR)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_model()
