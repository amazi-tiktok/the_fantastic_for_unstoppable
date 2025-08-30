from flask import Flask, render_template, request, jsonify
import re
from datetime import datetime
from policy import contains_commercial_info
from typing import Dict, List, Tuple, Optional

app = Flask(__name__)

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
        """Assess if review content is relevant to the store location"""
        if not review_text:
            return 0.8, ["Empty review text"]
            
        text_lower = review_text.lower()
        store_category = store_info.get('category', [''])[0].lower() if store_info.get('category') else ''
        store_name = store_info.get('name', '').lower()
        
        violations = []
        irrelevance_score = 0
        
        # Check for irrelevant content
        found_irrelevant = []
        for keyword in self.irrelevant_keywords:
            if keyword in text_lower:
                found_irrelevant.append(keyword)
                irrelevance_score += 0.25
        
        if found_irrelevant:
            violations.append(f"Contains irrelevant content: {', '.join(found_irrelevant)}")
        
        # Calculate relevance score based on store mentions
        relevance_score = 0
        
        # High relevance: mentions store category or specific name
        if store_category and store_category in text_lower:
            relevance_score += 0.6
        if store_name and any(word in text_lower for word in store_name.split() if len(word) > 3):
            relevance_score += 0.6
            
        # Medium relevance: generic location references
        location_words = ['place', 'here', 'location', 'store', 'business', 'shop']
        if any(word in text_lower for word in location_words):
            relevance_score += 0.3
        
        # Final irrelevance score (higher = more problematic)
        final_score = max(0, irrelevance_score - relevance_score)
        
        return min(final_score, 1.0), violations
    
    def analyze_visit_authenticity(self, review_text, rating):
        """Detect reviews from users who likely haven't visited the location"""
        if not review_text:
            return 0.5, ["Empty review - cannot verify visit"]
            
        text_lower = review_text.lower()
        violations = []
        suspicion_score = 0
        
        # Check for explicit non-visit indicators
        non_visit_found = []
        for indicator in self.non_visit_indicators:
            if indicator in text_lower:
                non_visit_found.append(indicator)
                suspicion_score += 0.7
        
        if non_visit_found:
            violations.append(f"Explicit non-visit indicators: {', '.join(non_visit_found)}")
        
        # Check for visit evidence (reduces suspicion)
        visit_evidence = sum(1 for indicator in self.visit_indicators if indicator in text_lower)
        
        # Suspicious patterns
        word_count = len(text_lower.split())
        
        # Long negative review with no visit evidence
        if rating <= 2 and visit_evidence == 0 and word_count > 15:
            violations.append("Lengthy negative review with no evidence of actual visit")
            suspicion_score += 0.5
        
        # Very short reviews are less suspicious for high ratings
        if rating >= 4 and word_count <= 10:
            suspicion_score *= 0.3
        
        # Visit evidence reduces suspicion
        if visit_evidence > 0:
            suspicion_score *= max(0.2, 1 - (visit_evidence * 0.3))
        
        return min(suspicion_score, 1.0), violations
    
    def calculate_quality_score(self, review_data, store_info):
        """Assess overall review quality"""
        text = review_data.get('text', '')
        rating = review_data.get('rating', 3)
        
        violations = []
        quality_issues = 0
        
        # Length checks
        word_count = len(text.split()) if text else 0
        if word_count < 3:
            violations.append("Review too short to be meaningful")
            return 0.8, violations
        
        if word_count == 1:
            violations.append("Single-word review")
            return 0.9, violations
        
        # Generic content detection
        generic_phrases = [
            'good place', 'nice location', 'okay place', 'fine', 
            'average', 'not bad', 'decent', 'alright'
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
        if generic_count > 0 and word_count < 8:
            violations.append("Generic or low-effort review")
            quality_issues += 0.4
        
        # Repetitive content
        words = text.lower().split()
        if len(set(words)) < len(words) * 0.5 and word_count > 5:
            violations.append("Highly repetitive content")
            quality_issues += 0.3
        
        # All caps (shouting)
        if text.isupper() and len(text) > 10:
            violations.append("Excessive use of capital letters")
            quality_issues += 0.2
        
        return min(quality_issues, 1.0), violations
    
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
            
            # Category-specific checks based on URL patterns
            store_category = store_info.get('category', ['Other'])[0]
            if store_category == 'Restaurant' and any(term in url_lower for term in ['car', 'vehicle', 'electronics']):
                violations.append(f"Image URL suggests content irrelevant to {store_category}")
                score += 0.4
            
        except Exception as e:
            violations.append(f"Error analyzing image URL: {str(e)}")
            score = 0.3
        
        return min(score, 1.0), violations if violations else ["Image URL passes basic checks"]
    
    def analyze_review(self, review_data, store_info):
        """Main analysis function that combines all policy checks including image analysis"""
        text = review_data.get('text', '')
        rating = review_data.get('rating', 3)
        reviewer_name = review_data.get('name', 'Anonymous')
        image_url = review_data.get('image_url', '')
        
        # Run all policy analyses
        ad_score, ad_violations = self.analyze_advertisement(text)
        relevancy_score, relevancy_violations = self.analyze_relevancy(text, store_info)
        visit_score, visit_violations = self.analyze_visit_authenticity(text, rating)
        quality_score, quality_violations = self.calculate_quality_score(review_data, store_info)
        image_score, image_violations = self.analyze_image(image_url, store_info)
        
        # Adjust weights based on whether image is provided
        if image_url:
            # When image is provided, use these weights
            weights = {
                'advertisement': 0.25,
                'relevancy': 0.2, 
                'visit_authenticity': 0.2,
                'quality': 0.15,
                'image_analysis': 0.2
            }
        else:
            # When no image, redistribute the weight
            weights = {
                'advertisement': 0.3,
                'relevancy': 0.25, 
                'visit_authenticity': 0.25,
                'quality': 0.2,
                'image_analysis': 0.0
            }
        
        overall_score = (
            ad_score * weights['advertisement'] +
            relevancy_score * weights['relevancy'] + 
            visit_score * weights['visit_authenticity'] +
            quality_score * weights['quality'] +
            image_score * weights['image_analysis']
        )
        
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
            }
        }
        
        # Only add image analysis if image URL was provided
        if image_url:
            policy_violations['image_analysis'] = {
                'score': round(image_score, 3),
                'violations': image_violations if image_violations else ["Image URL passes basic checks"],
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)