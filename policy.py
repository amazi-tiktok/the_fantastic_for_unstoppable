# from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re
from transformers import pipeline,AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO
from typing import Tuple

# change to a better model, here it is used because of desktop limitation
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# Load the small NSFW detection model once (fast and light)
nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
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

def classify_review_image_with_category(review_text: str, store_category: str):
    """
    Classify the review text with the store category and description.
    """
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

def detect_insulting_content(review_text: str) -> Tuple[float, list]:
    """
    Uses zero-shot classification to detect if the review is insulting or offensive.
    Returns a score (1 = very likely insulting, 0 = not insulting) and a list of violations.
    """
    result = classifier(
        review_text,
        candidate_labels=["insulting", "offensive", "polite", "neutral"],
        hypothesis_template="This review is {}."
    )
    # Take the highest score for "insulting" or "offensive", it's value would be between 0 and 1, 1 means very likely insulting
    insult_score = max(result['scores'][result['labels'].index("insulting")],
                       result['scores'][result['labels'].index("offensive")])
    violations = []
    if insult_score > 0.5:
        violations.append("Review contains insulting or offensive language")
    # we normalize the score, to avoid being too sensitive, 0.5 shouldn't be flagged?
    return insult_score, violations

def detect_sentiment(review_text: str):
    result = sentiment_analyzer(review_text)[0]
    return result['label'], result['score']

def classify_review_text_with_category(review_text: str, store_category: str, store_description: str) -> Tuple[float, list]:
    candidate = f"{store_category}: {store_description}" if store_description else store_category
    result = classifier(
        review_text,
        candidate_labels=[candidate, "other"],
        hypothesis_template="This review is about {}."
    )
    violations = []
    if result['scores'][0] < 0.8:
        violations.append("Low confidence in relevance between review and category/description")
    return 1 - result['scores'][0], violations