import pickle
import random
import argparse
from collections import defaultdict
import pandas as pd
import os
import re
import csv
import sys

#  Templates 
USER_TEMPLATES = [
    "User {user_id} exhibits a distinctive taste profile: {user_desc}.",
    "Based on past interactions, User {user_id}'s preferences can be summarized as: {user_desc}.",
    "User {user_id}'s culinary journey reveals a penchant for: {user_desc}.",
    "Analyzing historical data, we characterize User {user_id} as: {user_desc}.",
    "User {user_id}'s interests are best encapsulated by: {user_desc}.",
    "The interaction patterns of User {user_id} suggest a strong affinity for: {user_desc}.",
    "User {user_id} is recognized for favoring: {user_desc} in their culinary choices.",
    "Observing past ratings, User {user_id} appears to be drawn to: {user_desc}.",
    "User {user_id}'s profile distinctly reflects interests such as: {user_desc}.",
    "User {user_id} can be effectively described as having a taste for: {user_desc}.",
    "From previous interactions, we deduce that User {user_id} is inclined towards: {user_desc}.",
    "User {user_id}'s engagement history highlights a preference for: {user_desc}.",
    "A close look at User {user_id}'s behavior indicates they are: {user_desc}.",
    "User {user_id}'s historical choices strongly point to: {user_desc}.",
    "Based on their ratings, User {user_id} is best characterized by: {user_desc}."
]

ITEM_TEMPLATES = [
    "Considering this user's tastes, {item_name} stands out as a promising option, described as: {item_desc}.",
    "{item_name} appears well-suited for this user, given its features: {item_desc}.",
    "For a user with such preferences, {item_name} might be an excellent fit, thanks to: {item_desc}.",
    "The attributes of {item_name}—notably: {item_desc}—align with this user's interests.",
    "Given the user's profile, {item_name} could be highly appealing, as it is characterized by: {item_desc}.",
    "{item_name} seems to meet the user's expectations, featuring qualities such as: {item_desc}.",
    "This item, {item_name}, may capture the user's attention due to its key aspects: {item_desc}.",
    "With attributes like {item_desc}, {item_name} is likely to resonate with the user's tastes.",
    "The features of {item_name}—described by: {item_desc}—suggest it could match the user's interests.",
    "Based on the description, {item_name} is a plausible recommendation for this user, as it highlights: {item_desc}.",
    "The characteristics of {item_name}, notably {item_desc}, indicate it might suit the user's palate.",
    "Taking into account the user's preferences, {item_name}—which is noted for: {item_desc}—is a potential match.",
    "It is reasonable to expect that a user with these interests would appreciate {item_name}, defined by: {item_desc}.",
    "The profile of {item_name}, including elements like {item_desc}, aligns well with the user's taste.",
    "Given its description as: {item_desc}, {item_name} could be an appealing choice for this user."
]

#  Cleaning 
def clean_recipe_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return "unknown item"
    text = str(text)
    text = re.sub(r"[‘’`´]", "'", text)
    text = re.sub(r'[^a-zA-Z0-9\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "unknown item"

# Unified Data Creation Function 
def create_pairs(
        ratings, user_desc, item_desc, item_names,
        item_map,       # Internal Item ID -> Raw Item ID (map_i_n2o)
        user_map,       # Internal User ID -> Raw User ID (map_u_n2o) 
        user_template_list, item_template_list,
        negatives_per_positive=1,negative_seed=33):

    neg_rng = random.Random(negative_seed)
    positive_pairs = []
    negative_pairs = []
    user_interactions = defaultdict(set)

    # Pre-calculate interactions
    for item_id, user_dict in ratings.items():
        for user_id in user_dict.keys():
            user_interactions[user_id].add(item_id)

    all_item_ids = set(ratings.keys())

    # --- MAIN LOOP ---
    for item_id, user_dict in ratings.items():
        # 1. Map Internal Item ID -> Raw ID
        raw_item_id = item_map[item_id] 
        
        # 2. Get Item Data using Raw ID
        item_name = item_names.get(raw_item_id, "unknown item")
        item_desc_text = item_desc.get(raw_item_id, "No description available")

        for user_id in user_dict.keys():
            # 3. Map Internal User ID -> Raw ID
            raw_user_id = user_map[user_id] 

            # 4. Get User Data using Raw ID
            user_desc_text = user_desc.get(raw_user_id, "No description available")

            # --- POSITIVE PAIR ---
            user_template = random.choice(user_template_list)
            sentence_a = user_template.format(user_id=user_id, user_desc=user_desc_text)

            item_template = random.choice(item_template_list)
            sentence_b_pos = item_template.format(item_name=item_name, item_desc=item_desc_text)
            
            positive_pairs.append((sentence_a, sentence_b_pos, 1))

            # --- NEGATIVE PAIR ---
            non_interacted = list(all_item_ids - user_interactions[user_id])
            if not non_interacted:
                continue 
            sampled_negatives = neg_rng.sample(non_interacted, min(negatives_per_positive, len(non_interacted)))
            for neg_item_id in sampled_negatives:                
                # 5. Map Negative Item Internal -> Raw
                if neg_item_id not in item_map:
                    continue
                raw_neg_item_id = item_map[neg_item_id]

                # 6. Get Negative Item Data using Raw ID
                neg_item_name = item_names.get(raw_neg_item_id, "unknown item")
                neg_item_desc_text = item_desc.get(raw_neg_item_id, "No description available")

                neg_item_template = random.choice(item_template_list)
                sentence_b_neg = neg_item_template.format(item_name=neg_item_name, item_desc=neg_item_desc_text)
                
                negative_pairs.append((sentence_a, sentence_b_neg, 0))

    return positive_pairs, negative_pairs

def main():
    parser = argparse.ArgumentParser(description="Generate SPC datasets.")
    parser.add_argument('--dataset', type=str, required=True, choices=['foodcom', 'allrecipe'])
    parser.add_argument('--base_path', type=str, default="/data/nilu/coldreciperec/data")
    args = parser.parse_args()

    # --- 1. Configuration  ---
    if args.dataset == 'foodcom':
        print("--- Configuring for Food.com ---")
        data_dir = os.path.join(args.base_path, 'foodcom')
        metadata_path = os.path.join(data_dir, "metadata.pkl")
        
        # Point to the NEW recovered files
        user_desc_path = os.path.join(data_dir, "user_descriptions_RAW_ID.pkl")
        item_desc_path = os.path.join(data_dir, "item_descriptions_train_RAW_ID.pkl")
        val_item_desc_path = os.path.join(data_dir, "item_descriptions_val_RAW_ID.pkl")
        
        recipe_csv_path = os.path.join(data_dir, "RAW_recipes.csv")
        item_id_col = "id"
        item_name_col = "name"

    elif args.dataset == 'allrecipe':
        print("--- Configuring for AllRecipes ---")
        data_dir = os.path.join(args.base_path, 'allrecipe')
        metadata_path = os.path.join(data_dir, "metadata.pkl")
        
        user_desc_path = os.path.join(data_dir, "user_descriptions_RAW_ID.pkl")
        item_desc_path = os.path.join(data_dir, "item_descriptions_train_RAW_ID.pkl")
        val_item_desc_path = os.path.join(data_dir, "item_descriptions_val_RAW_ID.pkl")
        
        recipe_csv_path = os.path.join(data_dir, "raw-data_recipe.csv") 
        item_id_col = "recipe_id"
        item_name_col = "recipe_name"


    # --- 2. Load Metadata & Descriptions ---
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    train_i_ratings = metadata["train_i_ratings"]
    val_i_ratings = metadata["val_i_ratings"]
    
    # Load BOTH maps
    map_i_n2o = metadata["map_i_n2o"] # Item Internal -> Raw
    map_u_n2o = metadata["map_u_n2o"] # User Internal -> Raw

    print(f"Loading user descriptions from: {user_desc_path}")
    with open(user_desc_path, "rb") as f:
        user_descriptions = pickle.load(f)
    print(f"Loading item descriptions from: {item_desc_path}")
    with open(item_desc_path, "rb") as f:
        item_descriptions = pickle.load(f)
    print(f"Loading val item descriptions from: {val_item_desc_path}")
    with open(val_item_desc_path, "rb") as f:
        val_item_descriptions = pickle.load(f)

    # --- 3. Load CSV  ---
    print(f"Loading and cleaning recipe names from: {recipe_csv_path}")
    item_id_to_name = {}
    try:
        recipes_df = pd.read_csv(
            recipe_csv_path, 
            usecols=[item_id_col, item_name_col],
            dtype={item_id_col: int, item_name_col: str},
            on_bad_lines='skip',
            engine='python' 
        )
        for index, row in recipes_df.iterrows():
            try:
                r_id = int(row[item_id_col])
                r_name = row[item_name_col]
                item_id_to_name[r_id] = clean_recipe_text(r_name)
            except Exception:
                continue
        print(f"✅ Loaded names for {len(item_id_to_name)} items.")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")

    # --- 4. Select Templates & Run ---
    active_user_templates = USER_TEMPLATES
    active_item_templates = ITEM_TEMPLATES
    output_suffix = f"_{args.dataset}"
    

    print("Creating training dataset...")

    train_pos_pairs, train_neg_pairs = create_pairs(
        train_i_ratings, user_descriptions, item_descriptions,
        item_id_to_name, map_i_n2o, map_u_n2o,  
        active_user_templates, active_item_templates, negatives_per_positive=1, negative_seed=489
    )
    train_pairs = train_pos_pairs + train_neg_pairs
    random.shuffle(train_pairs)

    print("Creating validation dataset...")
    val_pos_pairs, val_neg_pairs = create_pairs(
        val_i_ratings, user_descriptions, val_item_descriptions,
        item_id_to_name, map_i_n2o, map_u_n2o,  # <--- Added map_u_n2o
        active_user_templates, active_item_templates
    )
    val_pairs = val_pos_pairs + val_neg_pairs
    random.shuffle(val_pairs)

    # --- 5. Save ---
    train_filename = f"sentence_pair_classification_data{output_suffix}_seed489.pkl"
    val_filename = f"sentence_pair_classification_val_data{output_suffix}_seed489.pkl"
    
    train_path = os.path.join(data_dir, train_filename)
    val_path = os.path.join(data_dir, val_filename)

    with open(train_path, "wb") as f:
        pickle.dump(train_pairs, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_pairs, f)

    print(f"\n✅ Data generation complete!")
    print(f"Train data saved to: {train_path} ({len(train_pairs)} pairs)")
    print(f"Val data saved to:   {val_path} ({len(val_pairs)} pairs)")

if __name__ == "__main__":
    main()