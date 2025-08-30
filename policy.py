# from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re


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

# -------------------------
# Step 2: NSFW image detection
# -------------------------

# clip_model_id = "openai/clip-vit-base-patch32"
# model = CLIPModel.from_pretrained(clip_model_id)
# processor = CLIPProcessor.from_pretrained(clip_model_id)

# # -------------------------
# # Step 2: Define NSFW / Safe prompts
# # -------------------------
# nsfw_prompts = [
#     "nude", 
#     "sexually explicit", 
#     "pornographic", 
#     "erotic"
# ]
# safe_prompts = [
#     "safe for work", 
#     "clothed", 
#     "normal photo", 
#     "non-explicit"
# ]


# # -------------------------
# # Step 3: CLIP-based NSFW detection
# # -------------------------
# def detect_nsfw_clip(image_path: str, threshold: float = 0.3) -> bool:
#     """
#     Returns True if the image is likely NSFW.
#     Uses CLIP similarity between image and textual prompts.
#     """
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(text=nsfw_prompts + safe_prompts, images=image, return_tensors="pt", padding=True)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # similarity = image_emb @ text_emb.T
#     image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
#     text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
    
#     # cosine similarity
#     similarity = torch.matmul(image_embeds, text_embeds.T).squeeze(0)
    
#     nsfw_scores = similarity[:len(nsfw_prompts)]
#     safe_scores = similarity[len(nsfw_prompts):]
    
#     nsfw_score = nsfw_scores.mean().item()
#     safe_score = safe_scores.mean().item()
    
#     # NSFW if NSFW similarity significantly higher than safe
#     return (nsfw_score - safe_score) > threshold

# # -------------------------
# # Step 1: Load image captioning model (BLIP-2)
# # -------------------------
# caption_model_id = "Salesforce/blip2-opt-2.7b"

# caption_processor = Blip2Processor.from_pretrained(caption_model_id)
# caption_model = Blip2ForConditionalGeneration.from_pretrained(
#     caption_model_id,
#     device_map="auto",
#     torch_dtype=torch.float16
# )

# def get_image_caption(image_path: str) -> str:
#     image = Image.open(image_path).convert("RGB")
#     inputs = caption_processor(images=image, return_tensors="pt").to(caption_model.device)
#     output = caption_model.generate(**inputs, max_new_tokens=50)
#     caption = caption_processor.decode(output[0], skip_special_tokens=True)
#     return caption

# # -------------------------
# # Test the captioner
# # -------------------------
# print(get_image_caption("food_photo.jpg"))

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