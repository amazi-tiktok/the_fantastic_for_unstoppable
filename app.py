from email.mime import text
from flask import Flask, render_template, request, jsonify
import re, csv, os
from datetime import datetime
from policy import contains_commercial_info, classify_review_image_with_category, detect_insulting_content, analyze_nsfw_content, detect_sentiment
from typing import Dict, List, Tuple, Optional

app = Flask(__name__)

CSV_FILE = 'feedback_log.csv'

class ReviewAnalyzer:
    def __init__(self):
        # Keywords for policy violation detection
        self.promo_keywords = [
            'discount', 'coupon', 'promo', 'sale', 'visit www', 
            'http', '.com', 'offer', 'deal', 'free shipping',
            'buy now', 'limited time', 'special offer'
        ]
        self.irrelevant_keywords = [
            'phone', 'car', 'weather', 'politics', 'movie', 
            'book', 'unrelated', 'my cat', 'video game'
        ]
        self.visit_indicators = [
            'went', 'visited', 'been here', 'came', 'arrived', 
            'inside', 'staff', 'service', 'ordered', 'bought'
        ]
        self.non_visit_indicators = [
            'never been', 'heard', 'someone told me', 
            'my friend said', 'read online'
        ]
        
        # Image analysis keywords for detecting promotional content
        self.image_promo_indicators = [
            'qr_code', 'barcode', 'website_url', 'discount_banner',
            'advertisement', 'promotional_poster', 'sale_sign'
        ]
        
        # Store category to expected image content mapping
        self.category_image_expectations = {
            'Restaurant': ['food', 'dining', 'menu', 'interior', 'dish', 'table', 'kitchen'],
            'Pharmacy': ['medicine', 'store_interior', 'pharmacy_counter', 'health_products'],
            'Retail': ['products', 'store_interior', 'merchandise', 'shopping'],
            'Gas Station': ['pump', 'station', 'fuel', 'convenience_store'],
            'Healthcare': ['medical', 'clinic', 'waiting_room', 'equipment'],
            'Automotive': ['car', 'vehicle', 'garage', 'parts', 'service_bay'],
            'Beauty': ['salon', 'spa', 'treatment', 'cosmetics', 'interior'],
            'Entertainment': ['venue', 'stage', 'screen', 'arcade', 'activity'],
            'Hotel': ['room', 'lobby', 'bed', 'amenities', 'building']
        }
    
    def analyze_advertisement(self, text):
        """Detect promotional content and advertisements"""
        if not text:
            return 0.0, []
            
        text_lower = text.lower()
        violations = []
        score = 0

        # Check for promotional keywords
        found_keywords = []
        for keyword in self.promo_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                score += 0.3

        if found_keywords:
            violations.append(f"Contains promotional keywords: {', '.join(found_keywords)}")

        # Check for URLs and web references
        if contains_commercial_info(text):
            violations.append("Contains URL or website reference")
            score += 0.5

        return min(score, 1.0), violations

    def analyze_relevancy(self, review_text, store_info):
        # """Assess if review content is relevant to the store category"""
        if not review_text:
            return 0.8, ["Empty review text"]
            
        store_category = store_info.get('category', [''])[0].lower() if store_info.get('category') else ''
        if store_category.strip() == 'others':
            return 0, []
        score, violations = classify_review_image_with_category(review_text, store_category)
        return min(score, 1.0), violations

    def analyze_visit_authenticity(self, username, review_text, rating):
        """Detect reviews from users who likely haven't visited the location"""
        if not review_text:
            return 0.5, ["Empty review - cannot verify visit"]
            
        violations = []
        score = 0.0

        # 1. Generic or repetitive text
        generic_phrases = [
            "great place", "nice service", "good experience", "highly recommend",
            "will come again", "very satisfied", "excellent", "awesome"
        ]
        text_lower = review_text.lower()
        is_generic = any(phrase in text_lower for phrase in generic_phrases)
        if is_generic:
            violations.append("Generic or template-like review text")
            score += 0.3

        # 2. Username pattern (random string or numbers)
        if username and re.match(r'^[a-zA-Z0-9]{8,}$', username):
            violations.append("Suspicious username pattern")
            score += 0.2

        # 5. Excessive punctuation or emojis (often used by bots)
        if review_text.count("!") > 3 or review_text.count("😊") > 2:
            violations.append("Excessive punctuation or emojis")
            score += 0.1

        # 6. Very short or very long reviews
        word_count = len(review_text.split())
        if word_count < 3 or word_count > 200:
            violations.append("Unusual review length")
            score += 0.1

        return min(score, 1.0), violations

    def analyze_quality(self, review_data, store_info):
        """Assess overall review quality"""
        text = review_data.get('text', '')
        rating = review_data.get('rating', 3)
        
        violations = []
        quality_issues = 0
        
        # Length checks
        word_count = len(text.split()) if text else 0
        if word_count == 0:
            return 0, violations

        word_count_score = min(word_count, 5)/5.0
        if word_count <= 5:
            violations.append("Review too short to be meaningful")
            return 1-word_count_score, violations

        # Generic content detection
        generic_phrases = [
            'good place', 'nice location', 'okay place', 'fine', 
            'average', 'not bad', 'decent', 'alright'
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
        if generic_count > 0 and word_count < 8:
            word_count_score = min(generic_count, 8)/8.0
            violations.append("Generic or low-effort review")
            quality_issues += (1 - word_count_score)/2

        # Repetitive content
        words = text.lower().split()
        if len(set(words)) < len(words) * 0.5 and word_count > 5:
            violations.append("Highly repetitive content")
            quality_issues += 0.1
        
        # All caps (shouting)
        if text.isupper() and len(text) > 10:
            violations.append("Excessive use of capital letters")
            quality_issues += 0.1

        # Sentiment-rating consistency
        # Get sentiment label and score
        sentiment_label, sentiment_score = detect_sentiment(text)

        # Check for mismatch between sentiment and rating
        if (sentiment_label == "POSITIVE" and rating <= 2) or (sentiment_label == "NEGATIVE" and rating >= 4):
            violations.append(
                f"Sentiment ({sentiment_label}) does not match star rating ({rating})"
            )
            quality_issues += 0.1
        return min(quality_issues, 1.0), violations

    def analyze_offensive(self, review_data):
        text = review_data.get('text', '')
        offensive_score, offensive_violations = detect_insulting_content(text)
        return offensive_score, offensive_violations

    def analyze_image(self, image_url: Optional[str], store_info: Dict) -> Tuple[float, List[str]]:
        """
        Analyze review image URL for policy violations
        
        Args:
            image_url: URL of the review image
            store_info: Store information including category
            
        Returns:
            Tuple of (violation_score, list_of_violations)
        """
        violations = []
        score = 0.0
        
        # If no image provided, return neutral score
        if not image_url or image_url == '':
            return 0.0, []
        
        try:
            # Analyze the URL itself for suspicious patterns
            url_lower = image_url.lower()
            
            # Check for promotional content indicators in URL
            promo_url_indicators = ['ad', 'banner', 'promo', 'sale', 'discount', 'coupon', 'offer']
            found_promo = [ind for ind in promo_url_indicators if ind in url_lower]
            if found_promo:
                violations.append(f"URL suggests promotional content: {', '.join(found_promo)}")
                score += 0.6
            
            # Check for stock photo services
            stock_services = ['shutterstock', 'getty', 'istock', 'unsplash', 'pexels', 'pixabay', 'stock']
            found_stock = [service for service in stock_services if service in url_lower]
            if found_stock:
                violations.append(f"Image from stock photo service: {', '.join(found_stock)}")
                score += 0.5
            
            # Check if URL is valid
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(image_url):
                violations.append("Invalid or suspicious image URL format")
                score += 0.3
            
            # Check for suspicious file extensions
            suspicious_extensions = ['.exe', '.zip', '.rar', '.bat', '.cmd', '.scr']
            if any(ext in url_lower for ext in suspicious_extensions):
                violations.append("Suspicious file extension in URL")
                score += 0.8
            
            # Check if it appears to be an image file
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
            has_image_ext = any(ext in url_lower for ext in image_extensions)
            
            # If URL doesn't end with image extension and doesn't look like a CDN/image service
            if not has_image_ext and not any(service in url_lower for service in ['imgur', 'cloudinary', 'picsum', 'placeholder']):
                violations.append("URL doesn't appear to point to an image file")
                score += 0.2
            
            """Analyze the text for severe causes of concern."""
            # check for image content, whether it is NSFW or not
            nsfw_score = analyze_nsfw_content(image_url)
            if nsfw_score > 0.5:
                violations.append("Image content is not appropriate")
                score += 0.8

        except Exception as e:
            violations.append(f"Error analyzing image URL: {str(e)}")
            score = 0.3
        
        return min(score, 1.0), violations

    def analyze_review(self, review_data, store_info):
        """Main analysis function that combines all policy checks including image analysis"""
        text = review_data.get('text', '')
        rating = review_data.get('rating', 3)
        reviewer_name = review_data.get('name', 'Anonymous')
        image_url = review_data.get('image_url', '')
        
        # Run all policy analyses
        ad_score, ad_violations = self.analyze_advertisement(text)
        relevancy_score, relevancy_violations = self.analyze_relevancy(text, store_info)
        visit_score, visit_violations = self.analyze_visit_authenticity(reviewer_name, text, rating)
        quality_score, quality_violations = self.analyze_quality(review_data, store_info)
        image_score, image_violations = self.analyze_image(image_url, store_info)

        print("Image analysis score:", image_score, image_violations)
        
        offensive_score, offensive_violations = self.analyze_offensive(review_data)
        # Adjust weights based on whether image is provided
        if image_url:
            # When image is provided, use these weights
            weights = {
                'advertisement': 0.2,
                'relevancy': 0.15, 
                'visit_authenticity': 0.1,
                'quality': 0.1,
                'image_analysis': 0.15,
                'offensive': 0.3
            }
        else:
            # When no image, redistribute the weight
            weights = {
                'advertisement': 0.2,
                'relevancy': 0.2, 
                'visit_authenticity': 0.2,
                'quality': 0.1,
                'image_analysis': 0.0,
                'offensive': 0.3
            }

        overall_score = (
            ad_score * weights['advertisement'] +
            relevancy_score * weights['relevancy'] + 
            visit_score * weights['visit_authenticity'] +
            quality_score * weights['quality'] +
            image_score * weights['image_analysis'] +
            offensive_score * weights['offensive']
        )
        
        overall_score = max(
            ad_score,
            relevancy_score,
            visit_score,
            image_score,
            offensive_score,
            overall_score  # (from weighted sum)
        )
        # removed quality score, it is less important in general

        # Determine recommended action based on score thresholds
        if overall_score >= 0.7:
            action = "REMOVE"
            action_reason = "High violation score - likely policy violation"
        elif overall_score >= 0.4:
            action = "FLAG"
            action_reason = "Medium violation score - requires manual review"
        else:
            action = "APPROVE"
            action_reason = "Low violation score - appears legitimate"
        
        # Only include image_analysis in results if image was provided
        policy_violations = {
            'advertisement': {
                'score': round(ad_score, 3),
                'violations': ad_violations,
                'weight': weights['advertisement']
            },
            'relevancy': {
                'score': round(relevancy_score, 3),
                'violations': relevancy_violations,
                'weight': weights['relevancy']
            },
            'visit_authenticity': {
                'score': round(visit_score, 3),
                'violations': visit_violations,
                'weight': weights['visit_authenticity']
            },
            'quality': {
                'score': round(quality_score, 3),
                'violations': quality_violations,
                'weight': weights['quality']
            },
            'offensive': {
                'score': round(offensive_score, 3),
                'violations': offensive_violations,
                'weight': weights['offensive']
            }
        }
        
        # Only add image analysis if image URL was provided
        if image_url:
            policy_violations['image_analysis'] = {
                'score': round(image_score, 3),
                'violations': image_violations if image_violations else [],
                'weight': weights['image_analysis']
            }

        return {
            'overall_violation_score': round(overall_score, 3),
            'action': action,
            'action_reason': action_reason,
            'policy_violations': policy_violations,
            'metadata': {
                'reviewer_name': reviewer_name,
                'review_length': len(text.split()) if text else 0,
                'rating': rating,
                'has_image': bool(image_url),
                'image_url': image_url if image_url else 'No image',
                'store_name': store_info.get('name', 'Unknown'),
                'store_category': store_info.get('category', ['Unknown'])[0] if store_info.get('category') else 'Unknown',
                'analysis_timestamp': datetime.now().isoformat(),
                'weights_used': weights
            }
        }

# Initialize the analyzer
analyzer = ReviewAnalyzer()

@app.route('/')
def index():
    """Render the main demo page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for review analysis"""
    try:
        # Parse incoming JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        review_data = data.get('review_data', {})
        store_info = data.get('store_info', {})
        gcom_id = data.get('gcom_id', 'N/A')
        
        # Basic validation
        if not review_data.get('text'):
            return jsonify({'error': 'Review text is required'}), 400
        
        if not store_info.get('name'):
            return jsonify({'error': 'Store name is required'}), 400
        
        # Perform the analysis
        results = analyzer.analyze_review(review_data, store_info)
        results['gcom_id'] = gcom_id
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.2.0'
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    feedback = data.get('feedback')
    results = data.get('results')
    if not feedback or not results:
        return jsonify({'error': 'Missing feedback or results'}), 400

    # Prepare row: timestamp, feedback, and each top-level key in results
    file_exists = os.path.isfile(CSV_FILE)

    # Get all top-level keys in results (order is preserved in Python 3.7+)
    result_keys = list(results.keys())
    header = ['timestamp', 'feedback'] + result_keys

    # Prepare row values
    row = [datetime.now().isoformat(), feedback] + [results.get(k, '') for k in result_keys]

    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Delete the feedback_log.csv file if it exists at startup, to avoid appending data to the old file, it may end up mixing the benchmark result for different model
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    app.run(debug=True, host='0.0.0.0', port=5000)