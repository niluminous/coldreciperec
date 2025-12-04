import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import math
import os
import time
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from transformers import logging
logging.set_verbosity_error()

# ------------------------------- #
# 1. Dataset Class
# ------------------------------- #
class SCPDataset(Dataset):
    def __init__(self, test_pairs, tokenizer, max_len):
        self.test_pairs = test_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.test_pairs)
    
    def __getitem__(self, idx):
        user_id, item_id, sentence_a, sentence_b, label = self.test_pairs[idx]
        
        sentence_a = str(sentence_a)
        sentence_b = str(sentence_b)
            
        encoding = self.tokenizer(
            sentence_a, sentence_b,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.tensor([])).squeeze(0), 
            "user_id": user_id,
            "item_id": item_id,
            "label": torch.tensor(label, dtype=torch.long)
        }

# ------------------------------- #
# 2. Vectorized Inference
# ------------------------------- #
@torch.no_grad()
def predict_scores(test_loader, model, device):
    user_item_scores = defaultdict(dict)
    model.eval()
    
    for batch in tqdm(test_loader, desc="Running Inference"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in batch and batch["token_type_ids"].nelement() > 0:
            if model.config.type_vocab_size > 1:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device, non_blocking=True)
        
        outputs = model(**inputs)
        
        # Get probability of class 1 (Positive Match)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        
        # Store results
        for i in range(len(user_ids)):
            u_id = user_ids[i].item() if isinstance(user_ids[i], torch.Tensor) else user_ids[i]
            i_id = item_ids[i].item() if isinstance(item_ids[i], torch.Tensor) else item_ids[i]
            user_item_scores[u_id][i_id] = probs[i]
            
    return user_item_scores
def cal_dcg(result, device):
    discount = torch.arange(len(result), device=device) + 2
    discount = torch.log2(discount.float())
    return (result / discount).sum()

def calculate_metrics_final(user_item_scores, ground_truth, all_test_items, K_LIST, device):
    """
    all_test_items: A list of ALL Item IDs that define the "Cold Start" candidate set.
    """
    result_dic_ndcg = {k: 0.0 for k in K_LIST}
    result_dic_recall = {k: 0.0 for k in K_LIST}
    
    # Pre-map Item ID 
    item_to_index = {item_id: idx for idx, item_id in enumerate(all_test_items)}
    num_test_items = len(all_test_items)
    
    valid_users = 0
    
    for user_id in tqdm(ground_truth.keys(), desc="Calculating Metrics"):
        if user_id not in user_item_scores:
            continue
        valid_users += 1
        
        # 1. Build Ground Truth Vector
        gt = torch.zeros(num_test_items, device=device)
        
        for item_id in ground_truth[user_id]:
            if item_id in item_to_index:
                gt[item_to_index[item_id]] = 1.0
        
        # 2. Build Prediction Vector 
        # Extract scores for the specific test items in order
        scores = []
        user_preds = user_item_scores[user_id]
        
        scores = [user_preds.get(item_id, -float('inf')) for item_id in all_test_items]
        scores_tensor = torch.tensor(scores, device=device)
        
        # 3. Ranking (Exact ColdGPT Logic)
        max_k = max(K_LIST)
        
        # Get Top-K indices from predictions
        _, indices = torch.topk(scores_tensor, k=max_k)
        
        # Get Ideal Top-K from GT (for IDCG)
        gt_values, _ = torch.topk(gt, k=max_k)
        
        for k in K_LIST:
            top_k_indices = indices[:k]
            result = gt[top_k_indices]
            
            # NDCG
            idcg = cal_dcg(gt_values[:k], device)
            dcg = cal_dcg(result, device)
            ndcg = dcg / (idcg + 1e-10)
            
            # Recall (Hits / Total Positives)
            total_positives = (gt != 0).sum()
            recall = (result != 0).sum() / (total_positives + 1e-10)
            
            result_dic_ndcg[k] += ndcg.item()
            result_dic_recall[k] += recall.item()

    # Average
    if valid_users > 0:
        for k in K_LIST:
            result_dic_ndcg[k] /= valid_users
            result_dic_recall[k] /= valid_users
            
    return result_dic_ndcg, result_dic_recall

# ------------------------------- #
# 3. Main Execution 
#------------------------------------#
def main():
    parser = argparse.ArgumentParser(description="Evaluate SPC Model using EXACT ColdGPT metrics.")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['foodcom', 'allrecipe'], help="Dataset name")
    parser.add_argument('--model_name', type=str, default="bert-base-uncased", help="Model architecture used")
    parser.add_argument('--base_path', type=str, default="/data/nilu/coldreciperec/data", help="Root data directory")
    parser.add_argument('--batch_size', type=int, default=2048, help="Inference batch size")
    parser.add_argument('--device', type=str, default="cuda:1", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Construct Paths 
    data_dir = os.path.join(args.base_path, args.dataset)
    
    # 1. Model Path
    # Matches the training run name: e.g., "bert-base-uncased_foodcom"
    model_run_name = f"{args.model_name.split('/')[-1]}_{args.dataset}" 
    
    # Matches the training file name: e.g., "best_model_foodcom_seed489f1"
    model_path = os.path.join(data_dir, "models", model_run_name, f"best_model_{args.dataset}")

    if not os.path.exists(model_path):
        model_path = os.path.join(data_dir, "models", model_run_name)
        
    # 2. Data Paths
    metadata_path = os.path.join(data_dir, "metadata.pkl")
    test_data_path = os.path.join(data_dir, f"scp_test_data_{args.dataset}.pkl")

    # 3. Output Paths 
    log_file_path = os.path.join(data_dir, "logs", model_run_name, "evaluation_log_COLDGPT_STYLE.txt")
    results_file_path = os.path.join(data_dir, "models", model_run_name, "evaluation_results_COLDGPT_STYLE.pkl")
    # --- Print Configuration ---
    print(f"--- Starting EXACT ColdGPT-style Evaluation ---")
    print(f"Dataset: {args.dataset}")
    print(f"Loading Model from: {model_path}")
    print(f"Loading Data from: {test_data_path}")
    
    # --- Load Resources ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at expected path: {model_path}")
            
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    print("Loading test data...")
    with open(test_data_path, "rb") as f:
        test_pairs = pickle.load(f)
        
    print("Loading metadata...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    print("Constructing Evaluation Sets...")
    
    # 1. Build Ground Truth from TEST PAIRS 
    ground_truth = defaultdict(set)
    test_items_in_pairs = set()
    
    for user_id, item_id, _, _, label in test_pairs:
        test_items_in_pairs.add(item_id)
        if label == 1:
            ground_truth[user_id].add(item_id)
            
    # 2. Define the Candidate Universe 
    all_test_items = sorted(list(test_items_in_pairs))
    
    # 3. Verification Step 
    metadata_test_items = set(metadata["test_i_ratings"].keys())
    missing_items = metadata_test_items - test_items_in_pairs
    

    # ... [Inference] ...
    test_dataset = SCPDataset(test_pairs, tokenizer, max_len=128)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    start_time = time.time()
    user_item_scores = predict_scores(test_loader, model, device)
    duration = time.time() - start_time
    
    # ... [Metrics] ...
    K_LIST = [5, 10, 20, 40]
    avg_ndcg, avg_recall = calculate_metrics_final(
        user_item_scores, 
        ground_truth, 
        all_test_items, 
        K_LIST, 
        device
    )

    print("\n" + "="*50)
    print(f"RESULTS ({args.dataset})")
    for k in K_LIST:
        print(f"Recall@{k}: {avg_recall[k]:.4f} | NDCG@{k}: {avg_ndcg[k]:.4f}")
    print("="*50)
    
# --- Display & Prepare Log ---
    output_str = [] 
    
    header = f"RESULTS ({args.dataset} - {args.mode})"
    print("\n" + "="*50)
    print(header)
    output_str.append("="*50)
    output_str.append(header)
    
    for k in K_LIST:
        line = f"Recall@{k}: {avg_recall[k]:.4f} | NDCG@{k}: {avg_ndcg[k]:.4f}"
        print(line)
        output_str.append(line)
        
    print("="*50)
    output_str.append("="*50)
    
    # --- SAVE RESULTS ---
    results = {
        "user_item_scores": user_item_scores, 
        "metrics": {"recall": avg_recall, "ndcg": avg_ndcg},
        "ground_truth": ground_truth,
        "inference_time_sec": duration,
        "config": {
            "dataset": args.dataset,
            "num_test_items": len(all_test_items)
        }
    }
    

    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    with open(results_file_path, "wb") as f:
        pickle.dump(results, f)
        
    # Save Text Log
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w") as f:
        f.write("\n".join(output_str))
        
    print(f"✅ Logs saved to: {log_file_path}")
    print(f"✅ Results object saved to: {results_file_path}")

if __name__ == "__main__":
    main()