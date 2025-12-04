import argparse
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import csv
import time
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hugging Face imports
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback
)

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.epoch_start_time = None
        self.train_only_duration = 0.0
        
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_time_sec', 
                'eval_loss', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall'
            ])

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            self.train_only_duration = time.time() - self.epoch_start_time

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            train_loss = "N/A"
            if state.log_history:
                for log in reversed(state.log_history):
                    if 'loss' in log:
                        train_loss = log['loss']
                        break

            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.epoch, 
                    train_loss,              
                    f"{self.train_only_duration:.2f}", 
                    metrics.get('eval_loss'),
                    metrics.get('eval_accuracy'),
                    metrics.get('eval_f1'),
                    metrics.get('eval_precision'),
                    metrics.get('eval_recall')
                ])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def load_pickle_to_hf_dataset(file_path):
    print(f"Loading data from: {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data, columns=["sentence1", "sentence2", "labels"]) 
    return Dataset.from_pandas(df)

def main():

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Fine-tune PLM on SPC data")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['foodcom', 'allrecipe'])
    parser.add_argument('--base_path', type=str, default="/data/nilu/coldreciperec/data")
    parser.add_argument('--model_name', type=str, default="bert-base-uncased")
    parser.add_argument('--batch_size', type=int, default=512) # Reduced batch size for safety?
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    

    suffix = f"_{args.dataset}"

    data_dir = os.path.join(args.base_path, args.dataset)
    train_file = os.path.join(data_dir, f"sentence_pair_classification_data{suffix}.pkl")
    val_file = os.path.join(data_dir, f"sentence_pair_classification_val_data{suffix}.pkl")
    run_name = f"{args.model_name.split('/')[-1]}_{args.dataset}"
    output_dir = os.path.join(data_dir, "models", run_name)
    logging_dir = os.path.join(data_dir, "logs", run_name)
    csv_log_path = os.path.join(output_dir, f"training_metrics_log{suffix}.csv")

    print(f"Output: {output_dir}")

    # --- Load Data ---
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    train_dataset = load_pickle_to_hf_dataset(train_file)
    val_dataset = load_pickle_to_hf_dataset(val_file)

    # --- Tokenization ---
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(
            examples['sentence1'], 
            examples['sentence2'], 
            truncation=True, 
            max_length=args.max_length
        )

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["sentence1", "sentence2"])
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["sentence1", "sentence2"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)

    config.hidden_dropout_prob = 0.1       
    config.attention_probs_dropout_prob = 0.1  
    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        config=config 
    )
    model.to("cuda") 

# --- Trainer ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=int(args.batch_size/2),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",      
        save_strategy="epoch",        
        load_best_model_at_end=True,  
        # label_smoothing_factor=0.1,  
        metric_for_best_model="f1",  
        greater_is_better=True,      
        # ------------------------
        
        save_total_limit=2,           
        logging_dir=logging_dir,
        logging_steps=50, 
        report_to="none",
        fp16=torch.cuda.is_available(), 
        dataloader_num_workers=4 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience), 
            CSVLoggerCallback(csv_log_path) 
        ]
    )

    # --- Train ---
    print("Starting Training...")
    start_total_time = time.time()
    train_result = trainer.train() 
    end_total_time = time.time()
    
    # --- Eval & Save ---
    eval_results = trainer.evaluate()
    eval_results["total_time_sec"] = end_total_time - start_total_time
    
    results_path = os.path.join(output_dir, f"final_eval_results{suffix}.txt")
    with open(results_path, "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
            
    final_model_path = os.path.join(output_dir, f"best_model{suffix}")
    trainer.save_model(final_model_path)
    print(f"Done. Saved to {final_model_path}")

if __name__ == "__main__":
    main()