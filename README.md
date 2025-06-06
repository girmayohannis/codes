import pandas as pd
import re
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer,AutoTokenizer, AutoModelForSequenceClassification,BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch

# Load and preprocess the dataset
file_path = '/content/es_train.csv'
df = pd.read_csv(file_path)

# Ensure the required columns exist
if 'text' not in df.columns or 'multiclass' not in df.columns:
    raise ValueError("The dataset must contain 'text' and 'multiclass' columns.")

# Drop missing values in critical columns
df = df.dropna(subset=['text', 'multiclass'])

# Map labels to integer values
label_map = {
    'Generalized Hope': 0,
    'Not Hope': 1,
    'Sarcasm': 2,
    'Realistic Hope': 3,
    'Unrealisticv Hope': 4
}
df['labels'] = df['multiclass'].map(label_map)

# Drop rows with unmapped labels (if any)
df = df.dropna(subset=['labels'])
df['labels'] = df['labels'].astype(int)  # Ensure labels are integers

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[@#&]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text.lower()

df['text'] = df['text'].apply(preprocess_text)

# Split dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('FacebookAI/xlm-roberta-large', num_labels=5)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(['text', 'multiclass', '__index_level_0__'])
test_dataset = test_dataset.remove_columns(['text', 'multiclass', '__index_level_0__'])

# Convert dataset to PyTorch tensors
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='macro'),
        "recall": recall_score(labels, preds, average='macro'),
        "f1": f1_score(labels, preds, average='macro')
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results_outputs",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the test set
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the model and tokenizer
trained_model_path = '/content/es_train_bert'
trainer.save_model(trained_model_path)
tokenizer.save_pretrained(trained_model_path)

print(f"Model and tokenizer saved at: {trained_model_path}")
