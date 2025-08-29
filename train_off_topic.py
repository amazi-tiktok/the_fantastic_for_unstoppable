import pandas as pd
import re
import os
import gzip
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import numpy as np # Needed for keyword extraction logic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# --- Configuration ---
METADATA_FILE = './data/meta-Alabama.json.gz' # Adjust if your file has a different name
REVIEWS_FILE = './data/review-Alabama.json.gz' # Adjust if your file has a different name
OUTPUT_DIR = './data/category_reviews' # Directory to save category-specific review OBJECT files
KEYWORDS_DIR = './data/category_keywords' # Directory to save extracted keywords
MAX_TFIDF_KEYWORDS = 35
TEXT_COLUMN_IN_REVIEWS = 'text'
CATEGORY_KEY_IN_METADATA = 'category' 
GMAP_ID_KEY_IN_METADATA = 'gmap_id'
GMAP_ID_KEY_IN_REVIEWS = 'gmap_id'
ALL_CATEGORIES_SUMMARY_FILE = 'all_categories_summary.json'
N_LINES_PREVIEW = 100 # Lines to read for initial inspection/progress
# Number of reviews to load for keyword extraction.
# For keyword extraction, we need a decent amount per category.
# Set this higher if your category files are empty after loading the first few lines.
N_LINES_FOR_KEYWORDS = 1000000 # Load up to 1000000 reviews per category for TF-IDF

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(KEYWORDS_DIR, exist_ok=True)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Helper Functions ---
def clean_text_for_tfidf(text):
    """Cleans text specifically for TF-IDF keyword extraction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def read_json_lines(filepath, limit=None):
    """Reads a gzipped JSON Lines file and returns a list of dictionaries,
       optionally limiting the number of records read."""
    data = []
    print(f"Reading file: {filepath} (limit={limit if limit else 'all'})")
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    if i % 10000 == 0: # Print progress reminder
                        print(f"  Reached reading limit of {limit} lines at line {i}. Stopping.")
                    break 
                if i > 0 and i % 50000 == 0: # General progress indicator for larger reads
                    print(f"  Processed {i} lines...")
                
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass # Silently skip bad lines
        print(f"Finished reading file. Loaded {len(data)} records.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def normalize_category_name(cat_name):
    """Cleans category name for use as filename/key."""
    if not isinstance(cat_name, str): return ""
    cat_name = cat_name.lower().replace(' ', '_').replace('&', 'and').replace('.', '')
    return re.sub(r'[^\w\-]+', '_', cat_name)

# --- Phase 1: Data Loading, Categorization, and Creating Category Review OBJECT Files ---
def process_data_for_categories_with_objects(metadata_filepath, reviews_filepath, output_dir, category_key, gmap_id_meta_key, gmap_id_review_key, text_review_key, review_limit_per_category_for_tf_idf, max_reviews_per_file):
    """
    Loads metadata and reviews, groups *entire review objects* by category,
    and saves them to category-specific JSON Lines files.
    """
    print("\n--- Phase 1: Processing Metadata and Reviews (Saving Full Objects) ---")

    # Load Metadata (load all for categorization)
    metadata = read_json_lines(metadata_filepath, limit=None) # Load all metadata
    if metadata is None: return None
    
    # Load Reviews (load a large chunk for keyword extraction)
    reviews_data = read_json_lines(reviews_filepath, limit=review_limit_per_category_for_tf_idf * 5) # Load a larger chunk of reviews
    if reviews_data is None: return None

    # Create DataFrames for easier manipulation
    metadata_df = pd.DataFrame(metadata)
    reviews_df = pd.DataFrame(reviews_data)

    # Filter metadata to get only necessary columns and clean category info
    metadata_df = metadata_df.dropna(subset=[gmap_id_meta_key, category_key]) # Remove rows missing essential keys
    
    # Ensure category is always a list
    def ensure_list_category(cat):
        if isinstance(cat, list):
            return cat
        elif isinstance(cat, str):
            return [cat] # Treat single string as a list
        else:
            return [] # Return empty list if not string or list

    metadata_df[category_key] = metadata_df[category_key].apply(ensure_list_category)
    
    # Explode the DataFrame to handle businesses with multiple categories easily
    # This will create a new row for each category a business belongs to.
    metadata_exploded = metadata_df.explode(category_key)
    
    # Remove rows where category is empty after explosion
    metadata_exploded = metadata_exploded[metadata_exploded[category_key].str.strip() != '']

    # Map gmap_id to reviews for efficient lookup
    reviews_by_gmap_id = reviews_df.groupby(gmap_id_review_key)

    # Create a dictionary to hold review OBJECTS per category
    # Structure: { 'category_name': [review_object1, review_object2, ...], ... }
    category_review_objects = defaultdict(list)

    print(f"Grouping review OBJECTS by category...")
    processed_meta_items = 0
    total_metadata_items = len(metadata_exploded) # Use exploded metadata for iteration

    # Iterate through the exploded metadata
    for _, row in metadata_exploded.iterrows():
        gmap_id = row.get(gmap_id_meta_key)
        category = row.get(category_key)

        if not gmap_id or not category: continue # Skip if essential info is missing

        # Normalize category name
        normalized_cat = category.lower().replace(' ', '_').replace('&', 'and').replace('.', '')
        if not normalized_cat: continue # Skip if category became empty after normalization

        # Find reviews for this gmap_id
        if gmap_id in reviews_by_gmap_id.groups:
            # Get the group of reviews for this gmap_id
            business_reviews_group = reviews_by_gmap_id.get_group(gmap_id)
            
            # Iterate through each review object for this business
            for _, review_item in business_reviews_group.iterrows():
                # Only add the review if it has the text field
                if text_review_key in review_item and review_item[text_review_key] and len(category_review_objects[normalized_cat]) < max_reviews_per_file:
                    # Pre-clean the text here if you want to save cleaned text
                    # For now, we save the raw object, cleaning is for TF-IDF later
                    category_review_objects[normalized_cat].append(review_item.to_dict())
        
        processed_meta_items += 1
        if processed_meta_items % 5000 == 0:
            print(f"  Processed {processed_meta_items}/{total_metadata_items} metadata entries...")

    print(f"\nFinished grouping review objects by category. Found {len(category_review_objects)} unique categories.")

    # Save entire review objects to separate JSON Lines files for each category
    print("Saving review OBJECTS to category-specific files...")
    category_files_map = {}
    for category, review_objects in category_review_objects.items():
        if not review_objects:
            continue
            
        safe_category_name = re.sub(r'[^\w\-]+', '_', category)
        category_filename = f"{safe_category_name}.json" # Use .json extension
        category_filepath = os.path.join(output_dir, category_filename)
        category_files_map[category] = category_filepath # Store path for Phase 2

        try:
            with open(category_filepath, 'w', encoding='utf-8') as f:
                json.dump(review_objects, f)
            print(f"  Saved {len(review_objects)} review objects for category '{category}' to {category_filepath}")
        except Exception as e:
            print(f"Error saving review objects for category '{category}': {e}")

    print("\nPhase 1 complete. Category-specific review OBJECT files created.")
    return category_files_map # Return dictionary of {category: filepath}


def process_and_merge_categories(category_files_folder, keywords_dir, num_target_categories, max_keywords_per_category, text_key_in_obj='text', num_reviews_for_tfidf=5000, bias_threshold=30):
    """
    Reads category JSONL files from a folder, extracts keywords for each,
    and then merges categories based on name+keywords using clustering.
    """
    skipped_categories = []
    if not os.path.isdir(category_files_folder):
        print(f"Error: Category files folder not found at '{category_files_folder}'. Please ensure it exists.")
        return None

    print(f"\n--- Processing Category Files from: {category_files_folder} ---")
    
    all_category_keywords = {} # Stores {category_name: [keywords]}
    category_files_map = {} # Store {category_name: filepath}

    # --- Part 1: Extract Keywords for each category ---
    print("--- Part 1: Extracting Keywords per Category ---")

    # Get all .json files from the specified folder
    category_filenames = [f for f in os.listdir(category_files_folder) if f.endswith('.json')]
    
    if not category_filenames:
        print(f"No '.json' files found in '{category_files_folder}'. Please check the folder.")
        return None
    
    print(f"Found {len(category_filenames)} category files.")

    for filename in category_filenames:
        filepath = os.path.join(category_files_folder, filename)
        # Filename itself is the category name (after removing extension)
        category_name = normalize_category_name(filename.replace('.json', ''))

        # Load review OBJECTS for TF-IDF
        category_data_objects = json.load(open(filepath))
        if len(category_data_objects) < bias_threshold:
            skipped_categories.append(category_name)
            print(f"  There are too little review objects in '{filename}'. Skipping.")
            continue
        review_texts_for_tfidf = []
        for obj in category_data_objects:
            review_text = obj.get(text_key_in_obj)
            if review_text:
                cleaned_text = clean_text_for_tfidf(review_text)
                if cleaned_text:
                    review_texts_for_tfidf.append(cleaned_text)
        
        if not review_texts_for_tfidf:
            print(f"  No valid review texts found for keyword extraction in '{category_name}'. Skipping.")
            continue
            
        print(f"  Loaded {len(review_texts_for_tfidf)} cleaned reviews for TF-IDF.")

        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(review_texts_for_tfidf)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            term_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            sorted_indices = np.argsort(term_scores)[::-1]
            
            top_keywords = [feature_names[i] for i in sorted_indices[:max_keywords_per_category]]
            all_category_keywords[category_name] = top_keywords # Store with original filename category name
            print(f"  Extracted top {len(top_keywords)} keywords: {top_keywords[:5]}...")

            # Save keywords
            safe_cat_name = normalize_category_name(category_name)
            keyword_filename = f"{safe_cat_name}_keywords.txt"
            keyword_filepath = os.path.join(keywords_dir, keyword_filename)
            with open(keyword_filepath, 'w', encoding='utf-8') as f:
                for kw in top_keywords: f.write(kw + '\n')
            print(f"  Saved keywords to {keyword_filepath}")

            # Store filepath for Phase 2 clustering
            category_files_map[category_name] = filepath 

        except Exception as e:
            print(f"Error processing keywords for category '{category_name}': {e}")

    if not all_category_keywords:
        print("No keywords were extracted for any category. Cannot proceed with merging.")
        return None

    # --- Part B: Cluster Categories based on Name + Keywords ---
    print(f"\n--- Part B: Clustering Categories into {num_target_categories} groups ---")

    category_names = list(all_category_keywords.keys())
    category_representations = []
    for category in category_names:
        keywords = all_category_keywords.get(category, [])
        # Create representation: Category Name + Keywords
        representation = f"{category.replace('_', ' ')} " + " ".join(keywords[:max_keywords_per_category])
        category_representations.append(representation)

    print(f"Created representations for {len(category_representations)} categories.")

    # Load Sentence-BERT model
    print("Generating Sentence-BERT embeddings for category representations...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        category_embeddings = model.encode(category_representations, show_progress_bar=True)
        print(f"Generated embeddings of shape: {category_embeddings.shape}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

    # Cluster the embeddings
    print(f"Clustering categories using Agglomerative Clustering into {num_target_categories} groups...")
    try:
        clustering_model = AgglomerativeClustering(
            n_clusters=num_target_categories,
            metric='cosine', 
            linkage='average'
        )
        cluster_labels = clustering_model.fit_predict(category_embeddings)
        print("Clustering complete.")
    except ValueError as ve:
        print(f"Clustering error: {ve}.")
        return None
    except Exception as e:
        print(f"Unexpected clustering error: {e}")
        return None

    # Group original categories by cluster labels and find representatives
    consolidated_categories = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        original_category = category_names[i]
        consolidated_categories[cluster_id].append(original_category)

    cluster_representatives = {}
    for cluster_id, original_categories in consolidated_categories.items():
        if not original_categories: continue
        
        indices_in_cluster = [category_names.index(cat) for cat in original_categories]
        embeddings_in_cluster = category_embeddings[indices_in_cluster]
        
        if len(embeddings_in_cluster) > 1:
            centroid = np.mean(embeddings_in_cluster, axis=0)
            distances = np.linalg.norm(embeddings_in_cluster - centroid, axis=1)
            closest_embedding_index_in_cluster = np.argmin(distances)
        else:
            closest_embedding_index_in_cluster = 0
        
        representative_category = original_categories[closest_embedding_index_in_cluster]
        clean_representative_name = representative_category.replace('_', ' ').title()
        
        cluster_representatives[cluster_id] = {
            "representative_name": clean_representative_name,
            "original_category_count": len(original_categories),
            "original_categories": original_categories
        }

    sorted_clusters = sorted(cluster_representatives.items(), key=lambda item: item[1]['original_category_count'], reverse=True)

    print(f"\n--- Top 10 Largest Consolidated Categories (out of {len(sorted_clusters)}) ---")
    for i, (cluster_id, data) in enumerate(sorted_clusters[:10]):
        print(f"{i+1}. Representative: '{data['representative_name']}' "
              f"(from {data['original_category_count']} original categories)")

    # Save the final mapping
    final_mapping = {
        "num_clusters": len(sorted_clusters),
        "clusters": [
            {
                "representative_name": data["representative_name"],
                "original_category_count": data["original_category_count"],
                "original_categories": data["original_categories"]
            }
            for cluster_id, data in sorted_clusters
        ],
        "skipped_categories": skipped_categories
    }
    
    output_filename = ALL_CATEGORIES_SUMMARY_FILE
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_mapping, f, indent=2)
        print(f"\nFinal consolidated category mapping saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving final category mapping: {e}")

    return final_mapping

# --- Main Execution ---
if __name__ == "__main__":
    current_dir = os.getcwd()
    metadata_path = os.path.join(current_dir, METADATA_FILE)
    reviews_path = os.path.join(current_dir, REVIEWS_FILE)
    cfg = {
        "execute_phase_1": False,
        "execute_phase_2": True,
    }
    # --- Phase 1 Execution ---
    if cfg["execute_phase_1"]:
      category_review_files_map = process_data_for_categories_with_objects(
          metadata_filepath=metadata_path,
          reviews_filepath=reviews_path,
          output_dir=OUTPUT_DIR,
          category_key=CATEGORY_KEY_IN_METADATA,
          gmap_id_meta_key=GMAP_ID_KEY_IN_METADATA,
          gmap_id_review_key=GMAP_ID_KEY_IN_REVIEWS,
          text_review_key=TEXT_COLUMN_IN_REVIEWS,
          review_limit_per_category_for_tf_idf=N_LINES_FOR_KEYWORDS,  # Pass the limit here
          max_reviews_per_file=1000  # Limit to 1000 reviews per category file
      )

    # open the json file inside ./data/category_reviews one by one, and dump reviews after 1000 reviews
    # dump_reviews_after_n(OUTPUT_DIR, NEW_OUTPUT_DIR, 1000)
    # --- Phase 2 Execution ---
    if cfg["execute_phase_2"]:
        result = process_and_merge_categories(OUTPUT_DIR, KEYWORDS_DIR, 100, MAX_TFIDF_KEYWORDS, TEXT_COLUMN_IN_REVIEWS, N_LINES_FOR_KEYWORDS)
        if result:
            print("Phase 2 executed successfully.")
        else:
            print("Phase 2 failed, please check the logs for more details.")
    else:
        print("Skipping Phase 2 as it is disabled in the configuration.")

    # here for phrase 3, we need to manually review the all_categories_summary.json file and correct it, in order to get an accurate representation of the categories and their relationships.
    # This may involve merging similar categories, renaming them for clarity, or removing duplicates.
    # The goal is to create a final, clean mapping of categories that can be used for training the model.

