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

# --- Configuration ---
METADATA_FILE = './data/meta-Alabama.json.gz' # Adjust if your file has a different name
REVIEWS_FILE = './data/review-Alabama.json.gz' # Adjust if your file has a different name
OUTPUT_DIR = './data/category_reviews' # Directory to save category-specific review OBJECT files
KEYWORDS_DIR = './data/category_keywords' # Directory to save extracted keywords
MAX_TFIDF_KEYWORDS = 30
TEXT_COLUMN_IN_REVIEWS = 'text'
CATEGORY_KEY_IN_METADATA = 'category' 
GMAP_ID_KEY_IN_METADATA = 'gmap_id'
GMAP_ID_KEY_IN_REVIEWS = 'gmap_id'
N_LINES_PREVIEW = 100 # Lines to read for initial inspection/progress
# Number of reviews to load for keyword extraction.
# For keyword extraction, we need a decent amount per category.
# Set this higher if your category files are empty after loading the first few lines.
N_LINES_FOR_KEYWORDS = 200000 # Load up to 200000 reviews per category for TF-IDF

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

# --- Phase 1: Data Loading, Categorization, and Creating Category Review OBJECT Files ---

def process_data_for_categories_with_objects(metadata_filepath, reviews_filepath, output_dir, category_key, gmap_id_meta_key, gmap_id_review_key, text_review_key, review_limit_per_category_for_tf_idf):
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
                if text_review_key in review_item and review_item[text_review_key]:
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
                for obj in review_objects:
                    json.dump(obj, f) # Dump the entire review object
                    f.write('\n')
            print(f"  Saved {len(review_objects)} review objects for category '{category}' to {category_filepath}")
        except Exception as e:
            print(f"Error saving review objects for category '{category}': {e}")

    print("\nPhase 1 complete. Category-specific review OBJECT files created.")
    return category_files_map # Return dictionary of {category: filepath}

# --- Phase 2: Keyword Extraction per Category ---

def extract_keywords_per_category(category_files_map, keywords_dir, max_keywords=30, text_key_in_obj='text'):
    """
    Loads category-specific review OBJECT files, extracts review text,
    and then extracts keywords using TF-IDF.
    """
    if not category_files_map:
        print("No category files provided for keyword extraction. Skipping Phase 2.")
        return {}

    print("\n--- Phase 2: Extracting Keywords per Category ---")
    all_category_keywords = {} # Store keywords for all categories

    for category, filepath in category_files_map.items():
        print(f"\nProcessing category for keywords: '{category}'")
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            print(f"  File not found or empty: {filepath}. Skipping keyword extraction.")
            continue

        # Load review OBJECTS for this category
        category_data_objects = read_json_lines(filepath, limit=N_LINES_FOR_KEYWORDS) # Use limit for keyword extraction
        if category_data_objects is None: continue
        
        # Extract JUST the cleaned text for TF-IDF
        review_texts_for_tfidf = []
        for obj in category_data_objects:
            review_text = obj.get(text_key_in_obj)
            if review_text:
                cleaned_text = clean_text_for_tfidf(review_text) # Clean specifically for TF-IDF
                if cleaned_text:
                    review_texts_for_tfidf.append(cleaned_text)
        
        if not review_texts_for_tfidf:
            print(f"  No valid review texts found for keyword extraction in category '{category}'. Skipping.")
            continue
            
        print(f"  Loaded {len(review_texts_for_tfidf)} cleaned reviews for TF-IDF.")

        # Initialize TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(review_texts_for_tfidf)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Calculate term scores (sum of TF-IDF across all docs for that term)
            term_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            
            # Get indices sorted by score (descending)
            sorted_indices = np.argsort(term_scores)[::-1]
            
            # Get the top N keywords
            top_keywords = [feature_names[i] for i in sorted_indices[:max_keywords]]
            
            all_category_keywords[category] = top_keywords
            print(f"  Extracted top {len(top_keywords)} keywords: {top_keywords}")

            # Save keywords to a file
            keyword_filename = f"{category}_keywords.txt"
            keyword_filepath = os.path.join(keywords_dir, keyword_filename)
            with open(keyword_filepath, 'w', encoding='utf-8') as f:
                for keyword in top_keywords:
                    f.write(keyword + '\n')
            print(f"  Saved keywords to {keyword_filepath}")

        except Exception as e:
            print(f"Error processing keywords for category '{category}': {e}")

    print("\nPhase 2 complete. Keywords extracted and saved.")
    return all_category_keywords

# --- Main Execution ---
if __name__ == "__main__":
    current_dir = os.getcwd()
    metadata_path = os.path.join(current_dir, METADATA_FILE)
    reviews_path = os.path.join(current_dir, REVIEWS_FILE)

    # --- Phase 1 Execution ---
    category_review_files_map = process_data_for_categories_with_objects(
        metadata_filepath=metadata_path,
        reviews_filepath=reviews_path,
        output_dir=OUTPUT_DIR,
        category_key=CATEGORY_KEY_IN_METADATA,
        gmap_id_meta_key=GMAP_ID_KEY_IN_METADATA,
        gmap_id_review_key=GMAP_ID_KEY_IN_REVIEWS,
        text_review_key=TEXT_COLUMN_IN_REVIEWS,
        review_limit_per_category_for_tf_idf=N_LINES_FOR_KEYWORDS # Pass the limit here
    )

    # --- Phase 2 Execution ---
    if category_review_files_map:
        all_extracted_keywords = extract_keywords_per_category(
            category_files_map=category_review_files_map,
            keywords_dir=KEYWORDS_DIR,
            max_keywords=MAX_TFIDF_KEYWORDS,
            text_key_in_obj=TEXT_COLUMN_IN_REVIEWS # Pass the key to get text from objects
        )
        
        print("\n--- Process Summary ---")
        print(f"Category review OBJECT files saved in: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Category keyword files saved in: {os.path.abspath(KEYWORDS_DIR)}")
        if all_extracted_keywords:
            print("\nKeywords extracted for categories (first 5 shown):")
            for cat, kws in all_extracted_keywords.items():
                print(f" - {cat}: {kws[:5]}...")
        else:
            print("No keywords were extracted.")
    else:
        print("\nSkipping Phase 2 due to issues in Phase 1.")