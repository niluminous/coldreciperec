# 🥗 Cold-Start Recipe Recommendation via Template-Based Language Inference with LLM-Generated Summaries


> **Official Implementation** of the paper submitted to *Expert Systems with Applications (ESWA)*.

## 📝 Introduction

Recipe-sharing platforms face the **Strict Cold-Start (SCS)** problem: newly uploaded recipes lack user interactions (ratings, clicks), making them invisible to traditional algorithms (CF, GNNs) that rely on interaction history.

In this work, we propose a novel framework that bridges **Large Language Models (LLMs)** and **Pre-trained Language Models (PLMs)** to solve this. Instead of relying on collaborative filtering, we reformulate recommendation as a **Sentence Pair Classification (SPC)** task.

By utilizing **GPT-4o-mini** to distill complex user and recipe attributes into coherent natural language summaries, we enable a BERT-based model to predict compatibility via **Natural Language Inference (NLI)**. Our approach achieves state-of-the-art performance on **SCS-Food.com** and **SCS-AllRecipes.com** benchmarks.

---

## 📂 Project Structure

Ensure your directory is organized as follows:

```text
coldreciperec/
├── data/                       # Main data directory
│   ├── foodcom/                # Food.com dataset files
│   │   ├── metadata.pkl
│   │   ├── user_descriptions_RAW_ID.pkl
│   │   ├── RAW_recipes.csv     # From Kaggle
│   │   └── ...
│   └── allrecipe/              # AllRecipes dataset files
│       ├── metadata.pkl
│       ├── raw-data_recipe.csv # From Kaggle
│       └── ...
├── make_train_data.py          # Generates training/validation pairs
├── make_test_data.py           # Generates ranking test data
├── finetune.py                 # Fine-tunes the BERT model
├── evaluate.py                 # Calculates NDCG & Recall metrics
└── README.md
```

## 📥 Data Setup

Before running the scripts, please download the required datasets from the sources below.

### 1. Download Datasets

| Dataset | Source | Description |
| :--- | :--- | :--- |
| **Summary Data** (Required) | [**Download via Figshare**](https://figshare.com/s/a25fefc9a8922899f733) | Contains the `foodcom` and `allrecipe` summary folders. |
| **Food.com Raw Data** | [**Download via Kaggle**](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) | Original recipe attributes and interaction data. |
| **AllRecipes Raw Data** | [**Download via Kaggle**](https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1) | Original recipe attributes and interaction data. |

### 2. File Placement

After downloading the **Summary Data** from Figshare, extract the contents and move the `foodcom` and `allrecipe` folders into your project directory here:

`coldreciperec/data/`

## 💻 Usage Guide


## Step 1: Generate Training Data
Create the training and validation datasets. This script handles negative sampling and template injection:

For Food.com
```bash
python make_train_data.py --dataset foodcom
```
For AllRecipes
```bash
python make_train_data.py --dataset allrecipe
```
## Step 2: Generate Test Data (Ranking)
Create the inference dataset. This generates a ranking list (User x All Items) formatted into natural language prompts.
```bash
python make_test_data.py --dataset foodcom 
```
Output: Saves scp_test_data_{dataset}.pkl.

## Step 3: Fine-tune the Model
Fine-tune the PLM (default: bert-base-uncased) on the sentence pairs.
```bash
python finetune.py \
    --dataset foodcom 
```
Outputs: Models are saved to ./data/{dataset}/models/.

## Step 4: Evaluation
Evaluate the model using NDCG@K and Recall@K .
```bash
python evaluate.py \
    --dataset foodcom 
```