from flask import Flask, render_template, request, jsonify
import json
import re
from datetime import datetime
import random

app = Flask(__name__)

class ReviewAnalyzer:
    def __init__(self):
        # Sample promotional keywords for demo
        self.promo_keywords = ['discount', 'coupon', 'promo', 'sale', 'visit www', 'http', '.com', 'offer', 'deal']
        self.irrelevant_keywords = ['phone', 'car', 'weather', 'politics', 'movie', 'book', 'unrelated']
        self.visit_indicators = ['went', 'visited', 'been here', 'came', 'arrived', 'inside', 'staff', 'service']
        self.non_visit_indicators = ['never been', 'heard', 'someone told me', 'my friend said']
    
    def analyze_advertisement(self, text):
        """Detect promotional content"""
        text_lower = text.lower()
        violations = []
        score = 0
        
        for keyword in self.promo_keywords:
            if keyword in text_lower:
                violations.append(f"Contains promotional keyword: '{keyword}'")
                score += 0.3
        
        # Check for URLs
        if re.search(r'www\.|http|\.com|\.org', text_lower):
            violations.append("Contains URL or website reference")
            score += 0.5
        
        return min(score, 1.0), violations
    
    def analyze_relevancy(self, review_text, store_info):
        """Assess if review is relevant to the location"""
        text_lower = review_text.lower()
        store_category = store_info.get('category', [''])[0].lower()
        store_name = store_info.get('name', '').lower()
        
        violations = []
        score = 0
        
        # Check for irrelevant content
        for keyword in self.irrelevant_keywords:
            if keyword in text_lower:
                violations.append(f"Contains irrelevant content: '{keyword}'")
                score += 0.2
        
        # Check if review mentions store category or name
        relevance_score = 0
        if store_category in text_lower or any(word in text_lower for word in store_name.split()):
            relevance_score = 0.8
        elif any(word in text_lower for word in ['place', 'here', 'location', 'store', 'business']):
            relevance_score = 0.5
        
        final_score = score - relevance_score
        return max(0, min(final_score, 1.0)), violations
    
    def analyze_visit_authenticity(self, review_text, rating):
        """Detect rants from users who likely haven't visited"""
        text_lower = review_text.lower()
        violations = []
        score = 0
        
        # Check for explicit non-visit indicators
        for indicator in self.non_visit_indicators:
            if indicator in text_lower:
                violations.append(f"Indicates no visit: '{indicator}'")
                score += 0.8
        
        # Check for visit indicators (reduces suspicion)
        visit_evidence = sum(1 for indicator in self.visit_indicators if indicator in text_lower)
        
        # Low rating with no visit evidence is suspicious
        if rating <= 2 and visit_evidence == 0 and len(text_lower.split()) > 10:
            violations.append("Low rating with no evidence of actual visit")
            score += 0.4
        
        # High rating is less suspicious
        if rating >= 4:
            score *= 0.5
        
        return min(score, 1.0), violations
    
    def calculate_quality_score(self, review_data, store_info):
        """Overall quality assessment"""
        text = review_data.get('text', '')
        rating = review_data.get('rating', 3)
        
        # Length check
        if len(text.split()) < 3:
            return 0.8, ["Review too short to be meaningful"]
        
        # Generic review detection
        generic_phrases = ['good place', 'nice location', 'okay', 'fine', 'average']
        if any(phrase in text.lower() for phrase in generic_phrases) and len(text.split()) < 8:
            return 0.6, ["Generic or low-effort review"]
        
        return 0.1, []
    
    def analyze_review(self, review_data, store_info):
        """Main analysis function"""
        text = review_data.get('text', '')
        rating = review_data.get('rating', 3)
        
        # Run all analyses
        ad_score, ad_violations = self.analyze_advertisement(text)
        relevancy_score, relevancy_violations = self.analyze_relevancy(text, store_info)
        visit_score, visit_violations = self.analyze_visit_authenticity(text, rating)
        quality_score, quality_violations = self.calculate_quality_score(review_data, store_info)
        
        # Calculate overall violation score
        overall_score = (ad_score + relevancy_score + visit_score + quality_score) / 4
        
        # Determine action
        if overall_score > 0.7:
            action = "REMOVE"
        elif overall_score > 0.4:
            action = "FLAG"
        else:
            action = "APPROVE"
        
        return {
            'overall_violation_score': round(overall_score, 3),
            'action': action,
            'policy_violations': {
                'advertisement': {
                    'score': round(ad_score, 3),
                    'violations': ad_violations
                },
                'relevancy': {
                    'score': round(relevancy_score, 3),
                    'violations': relevancy_violations
                },
                'visit_authenticity': {
                    'score': round(visit_score, 3),
                    'violations': visit_violations
                },
                'quality': {
                    'score': round(quality_score, 3),
                    'violations': quality_violations
                }
            },
            'metadata': {
                'review_length': len(text.split()),
                'rating': rating,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

analyzer = ReviewAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        review_data = data.get('review_data', {})
        store_info = data.get('store_info', {})
        gcom_id = data.get('gcom_id', 'N/A')
        
        # Perform analysis
        results = analyzer.analyze_review(review_data, store_info)
        results['gcom_id'] = gcom_id
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)