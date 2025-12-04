import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
import sys
import csv
import re

# --- Templates ---
USER_TEMPLATE_FIXED = "User {user_id} exhibits a distinctive taste profile: {user_desc}."
ITEM_TEMPLATE_FIXED = "For a user with such preferences, {item_name} might be an excellent fit, thanks to qualities such as: {item_desc}."

# CSV Name Loader ---
def load_recipe_names_robust(csv_path, id_col, name_col):
    print(f"Loading names from: {csv_path}")
    id_to_name = {}
    try:

        df = pd.read_csv(
            csv_path, 
            usecols=[id_col, name_col],
            dtype={id_col: int, name_col: str},
            on_bad_lines='skip',
            engine='python' 
        )
        for _, row in df.iterrows():
            try:
                r_id = int(row[id_col])
                r_name = str(row[name_col])
                # Clean text
                r_name = re.sub(r"[^a-zA-Z0-9\s']", ' ', r_name)
                r_name = re.sub(r'\s+', ' ', r_name).strip()
                if r_name:
                    id_to_name[r_id] = r_name
            except:
                continue
    except Exception as e:
        print(f"❌ Error loading names: {e}")
    
    print(f"✅ Loaded {len(id_to_name)} names.")
    return id_to_name

# --- Unified Data Creation Function ---
def create_test_pairs_ranking(
        test_ratings, user_desc, item_desc, item_names, 
        item_map, # Internal -> Raw Item
        user_map  # Internal -> Raw User 
    ):

    test_pairs = []
    test_users = set()
    for user_dict in test_ratings.values():
        test_users.update(user_dict.keys())
    test_users = list(test_users)
    
    test_items = list(test_ratings.keys())
    
    positive_interactions = set()
    for item_id, user_dict in test_ratings.items():
        for user_id in user_dict.keys():
            positive_interactions.add((user_id, item_id))
            
    print(f"Generating Ranking Data for {len(test_users)} Users x {len(test_items)} Items...")
    print(f"Total Pairs to Generate: {len(test_users) * len(test_items)}")

    # Generate Pairs (User x All Items)
    for user_id in tqdm(test_users, desc="Generating Pairs"):
        
        # --- PREPARE USER SENTENCE ---
        # Map Internal User ID -> Raw ID
        raw_user_id = user_map[user_id] 
        # Get Description using Raw ID
        u_desc_txt = user_desc.get(raw_user_id, "No description available")
        
        sentence_a = USER_TEMPLATE_FIXED.format(user_id=user_id, user_desc=u_desc_txt)
        
        for item_id in test_items:
            # --- PREPARE ITEM SENTENCE ---
            # Map Internal Item ID -> Raw ID
            raw_item_id = item_map[item_id]
            
            # Get Name using Raw ID
            i_name = item_names.get(raw_item_id, "Unknown Recipe")
            # Get Description using Raw ID
            i_desc_txt = item_desc.get(raw_item_id, "No description available")
            
            sentence_b = ITEM_TEMPLATE_FIXED.format(item_name=i_name, item_desc=i_desc_txt)
            # Label
            label = 1 if (user_id, item_id) in positive_interactions else 0
            
            # Format: (User_ID, Item_ID, Sent_A, Sent_B, Label)
            test_pairs.append((user_id, item_id, sentence_a, sentence_b, label))
            
    return test_pairs

def main():
    parser = argparse.ArgumentParser(description="Generate SCP test datasets.")
    parser.add_argument('--dataset', type=str, required=True, choices=['foodcom', 'allrecipe'])
    parser.add_argument('--base_path', type=str, default="/data/nilu/coldreciperec/data")
    args = parser.parse_args()

    # --- 1. Configuration  ---
    if args.dataset == 'foodcom':
        print("--- Configuring for Food.com ---")
        data_dir = os.path.join(args.base_path, 'foodcom')
        metadata_path = os.path.join(data_dir, "metadata.pkl")
        
        # Point to RAW ID files
        user_desc_path = os.path.join(data_dir, "user_descriptions_RAW_ID.pkl")
        test_item_desc_path = os.path.join(data_dir, "item_descriptions_test_RAW_ID.pkl")
        
        recipe_csv_path = os.path.join(data_dir, "RAW_recipes.csv")
        item_id_col = "id"
        item_name_col = "name"

    elif args.dataset == 'allrecipe':
        print("--- Configuring for AllRecipes ---")
        data_dir = os.path.join(args.base_path, 'allrecipe')
        metadata_path = os.path.join(data_dir, "metadata.pkl")
        
        # Point to RAW ID files
        user_desc_path = os.path.join(data_dir, "user_descriptions_RAW_ID.pkl")
        test_item_desc_path = os.path.join(data_dir, "item_descriptions_test_RAW_ID.pkl")
        
        recipe_csv_path = os.path.join(data_dir, "raw-data_recipe.csv")
        item_id_col = "recipe_id"
        item_name_col = "recipe_name"

    # --- 2. Load Data ---
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    test_i_ratings = metadata["test_i_ratings"]
    
    map_i_n2o = metadata["map_i_n2o"] # Item Internal -> Raw
    map_u_n2o = metadata["map_u_n2o"] # User Internal -> Raw

    print(f"Loading user descriptions from: {user_desc_path}")
    with open(user_desc_path, "rb") as f:
        user_descriptions = pickle.load(f)
        
    print(f"Loading TEST item descriptions from: {test_item_desc_path}")
    if os.path.exists(test_item_desc_path):
        with open(test_item_desc_path, "rb") as f:
            test_item_descriptions = pickle.load(f)
    else:
  
        fallback_path = os.path.join(data_dir, "item_descriptions_train_RAW_ID.pkl")
        print(f"⚠️ Test desc file not found. Falling back to: {fallback_path}")
        with open(fallback_path, "rb") as f:
            test_item_descriptions = pickle.load(f)

    item_id_to_name = load_recipe_names_robust(recipe_csv_path, item_id_col, item_name_col)

    # --- Generate Data ---
    scp_test_pairs = create_test_pairs_ranking(
        test_i_ratings,
        user_descriptions,
        test_item_descriptions,
        item_id_to_name,
        map_i_n2o,
        map_u_n2o 
    )

    # --- Statistics ---
    num_pos = sum(1 for p in scp_test_pairs if p[4] == 1)
    num_neg = len(scp_test_pairs) - num_pos
    
    print(f"\n✅ Generated {len(scp_test_pairs)} pairs.")
    print(f"   Positives: {num_pos}")
    print(f"   Negatives: {num_neg}")

    # --- Save ---
    output_filename = f"scp_test_data_{args.dataset}.pkl"
    output_path = os.path.join(data_dir, output_filename)
    
    print(f"Saving to {output_path}...")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(scp_test_pairs, f, protocol=4)
        print("✅ Saved successfully.")
    except Exception as e:
        print(f"❌ Error saving pickle: {e}")

if __name__ == "__main__":
    main()