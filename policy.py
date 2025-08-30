# from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re
from transformers import pipeline,AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

# change to a better model, here it is used because of desktop limitation
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# Load the small NSFW detection model once (fast and light)
nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

# -------------------------
# Step 0: Heuristic checks
# -------------------------
def contains_commercial_info(text: str) -> bool:
    phone_regex = r"\+?\d[\d\s-]{7,}\d"
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    # Check for URLs and web references
    contains_url = False
    url_patterns = [r'www\.', r'http[s]?://', r'\.com\b', r'\.org\b', r'\.net\b']
    for pattern in url_patterns:
        if re.search(pattern, text):
            contains_url = True
            break

    return bool(re.search(phone_regex, text) or re.search(email_regex, text)) or contains_url

def contains_profanity(text: str) -> bool:
    bad_words = ["damn", "shit", "fuck", "bitch"]
    return any(word in text.lower() for word in bad_words)

def classify_review_with_category(review_text: str, store_category: str):
    result = classifier(
        review_text,
        candidate_labels=[store_category, "other"],
        hypothesis_template="This review is about {}."
    )
    violations = []
    if result['scores'][0] < 0.8:
        violations.append("Low confidence in relevance between image and category")

    return 1- result['scores'][0], violations

# -------------------------
# Step 2: NSFW image detection
# -------------------------

def analyze_nsfw_content(image_path_or_url: str, threshold: float = 0.5) -> bool:
    """
    Returns True if the image is likely NSFW.
    Uses a small, efficient image classification model.
    """
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
    inputs = nsfw_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = nsfw_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        nsfw_prob = probs[0][1].item()  # Index 1 is NSFW, 0 is SFW
    return nsfw_prob

# -------------------------
# Step 2: Load text classifier (Mistral-7B-Instruct)
# -------------------------
# def classify_review_with_image(business_category: str, review: str) -> str:
#     classifier_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
#     tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
#     classifier_model = AutoModelForCausalLM.from_pretrained(
#         classifier_model_id,
#         device_map="auto",
#         dtype="auto"
#     )

#     classifier_pipeline = pipeline(
#         "text-generation",
#         model=classifier_model,
#         tokenizer=tokenizer,
#         max_new_tokens=128,
#         temperature=0.05
#     )

#     # caption = get_image_caption(image_path)

#     prompt = f"""
# You are a classifier.
# Task: Decide if the following Google Maps review text is ON-TOPIC or OFF-TOPIC 
# for the given business category and also the confident level between 0.0 and 1.0. Respond ONLY in JSON:

# {{"classification": "ON-TOPIC" or "OFF-TOPIC", "confidence": 0.0-1.0}}

# Business category: {business_category}
# Review text: "{review}"
# Answer:
# """
#     result = classifier_pipeline(prompt)
#     return result[0]["generated_text"]

# # -------------------------
# # Example usage
# # -------------------------
# print(contains_commercial_info("My phone number is +1234567890"))
# print(classify_review_with_image(
#     "Restaurant",
#     "The pizza was delicious and the service was friendly.",
#     # "food_photo.jpg"
# ))

# print(classify_review_with_image(
#     "Restaurant",
#     "I don't like the mayor of this city.",
#     # "random_city_photo.jpg"
# ))

# print(detect_nsfw_clip("food_photo.jpg"))