from flask import Flask, render_template, request, jsonify
import re
from datetime import datetime
from policy import contains_commercial_info

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
    
    def analyze_review(self, review_data, store_info):
        """Main analysis function that combines all policy checks"""
        text = review_data.get('text', '')
        rating = review_data.get('rating', 3)
        reviewer_name = review_data.get('name', 'Anonymous')
        
        # Run all policy analyses
        ad_score, ad_violations = self.analyze_advertisement(text)
        relevancy_score, relevancy_violations = self.analyze_relevancy(text, store_info)
        visit_score, visit_violations = self.analyze_visit_authenticity(text, rating)
        quality_score, quality_violations = self.calculate_quality_score(review_data, store_info)
        
        # Calculate weighted overall violation score
        weights = {
            'advertisement': 0.3,
            'relevancy': 0.25, 
            'visit_authenticity': 0.25,
            'quality': 0.2
        }
        
        overall_score = (
            ad_score * weights['advertisement'] +
            relevancy_score * weights['relevancy'] + 
            visit_score * weights['visit_authenticity'] +
            quality_score * weights['quality']
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
        
        return {
            'overall_violation_score': round(overall_score, 3),
            'action': action,
            'action_reason': action_reason,
            'policy_violations': {
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
            },
            'metadata': {
                'reviewer_name': reviewer_name,
                'review_length': len(text.split()) if text else 0,
                'rating': rating,
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
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)