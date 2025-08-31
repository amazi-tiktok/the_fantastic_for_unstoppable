import pandas as pd
import gzip
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# -----------------------------
# Step 1: Load dataset
# -----------------------------
file_path = "review-Alabama_10.json.gz"

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

print(df.head())
print("\nColumns in dataset:", df.columns)

# -----------------------------
# Step 2: Filter reviews with images
# -----------------------------
if 'pics' in df.columns:
    reviews_with_images = df[df['pics'].notnull()]
    print("\nTotal reviews with images:", len(reviews_with_images))
    print(reviews_with_images[['name', 'rating', 'text', 'pics']].head())
else:
    print("\nNo 'pics' column found in dataset. Please check columns above.")
    reviews_with_images = pd.DataFrame()  # empty fallback

# -----------------------------
# Step 3: Load CLIP model
# -----------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -----------------------------
# Step 4: Function to check relevance
# -----------------------------
def check_relevance(review_text, image_url):
    try:
        # Load image from URL
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Encode text & image
        inputs = processor(
            text=[review_text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = model(**inputs)

        # Extract embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity
        similarity = torch.matmul(text_embeds, image_embeds.T)
        score = similarity.item()

        return score  # -1 to 1 (closer to 1 = highly relevant)
    except Exception as e:
        print("Error checking relevance:", e)
        return None

# -----------------------------
# Step 5: Run relevance check
# -----------------------------
sample_reviews = reviews_with_images.head(20)  # check first 20 reviews
for idx, row in sample_reviews.iterrows():
    text = row['text']
    if not isinstance(text, str) or text.strip() == "":
        print("\nSkipping empty or invalid review")
        continue

    # Extract actual URL from 'pics'
    img_url = None
    if isinstance(row['pics'], list) and isinstance(row['pics'][0], dict):
        img_url = row['pics'][0]['url'][0]  # first URL
    elif isinstance(row['pics'], str):
        img_url = row['pics']

    if img_url:
        try:
            text_truncated = text[:200]  # limit to 200 chars
            score = check_relevance(text_truncated, img_url)
            print("\nReview:", text_truncated)
            print("Image URL:", img_url)
            print("Relevance Score:", score)
        except Exception as e:
            print("Error checking relevance:", e)
    else:
        print("\nReview:", text)
        print("No valid image URL found.")
