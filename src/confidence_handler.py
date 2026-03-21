# src/confidence_handler.py
"""
Day 5 - Task 5.3: Confidence Handler
Purpose: Add uncertainty quantification to predictions and recommendations
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

class ConfidenceHandler:
    """Handle prediction confidence and uncertainty flags"""
    
    def __init__(self):
        # Confidence thresholds
        self.thresholds = {
            'high': 0.75,    # 75%+ = high confidence
            'medium': 0.50,  # 50-75% = medium confidence
            'low': 0.25      # <50% = low confidence
        }
        
        # Uncertainty levels
        self.uncertainty_levels = {
            'low': {
                'label': 'Confident',
                'message': 'Strong recommendation based on clear patterns',
                'action': 'follow'
            },
            'medium': {
                'label': 'Moderate Confidence',
                'message': 'Suggestion based on available information',
                'action': 'consider'
            },
            'high': {
                'label': 'Uncertain',
                'message': 'Limited confidence. Use your judgment.',
                'action': 'evaluate'
            }
        }
        
        # Factors that increase uncertainty
        self.uncertainty_factors = {
            'short_text': 0.15,      # <=5 words
            'missing_data': 0.20,    # Missing key features
            'low_model_confidence': 0.25,  # Model prediction <50%
            'contradictory_signals': 0.20,  # Conflicting features
            'rare_emotion': 0.10     # Uncommon emotion pattern
        }
    
    def get_confidence_level(self, confidence_score):
        """Convert confidence score to level label"""
        
        if confidence_score >= self.thresholds['high']:
            return 'low'  # Low uncertainty = high confidence
        elif confidence_score >= self.thresholds['medium']:
            return 'medium'
        else:
            return 'high'  # High uncertainty = low confidence
    
    def calculate_prediction_confidence(self, model_proba, text_features=None):
        """Calculate overall prediction confidence"""
        
        # Base confidence from model probability
        base_confidence = float(np.max(model_proba))
        
        # Adjust for uncertainty factors
        uncertainty_penalty = 0.0
        
        # Check for short text
        if text_features:
            if text_features.get('is_short', False):
                uncertainty_penalty += self.uncertainty_factors['short_text']
            
            if text_features.get('ambiguity_score', 0) > 0.05:
                uncertainty_penalty += self.uncertainty_factors['contradictory_signals']
        
        # Calculate final confidence
        final_confidence = max(0.0, base_confidence - uncertainty_penalty)
        
        return {
            'confidence_score': round(final_confidence, 3),
            'confidence_percent': round(final_confidence * 100, 1),
            'uncertainty_level': self.get_confidence_level(final_confidence),
            'base_confidence': round(base_confidence, 3),
            'uncertainty_penalty': round(uncertainty_penalty, 3)
        }
    
    def get_recommendation_confidence(self, emotion_confidence, intensity_mae=None):
        """Calculate confidence for recommendation"""
        
        # Start with emotion prediction confidence
        rec_confidence = emotion_confidence
        
        # Adjust based on intensity prediction error (if available)
        if intensity_mae:
            if intensity_mae > 1.5:
                rec_confidence *= 0.8  # Reduce confidence for high MAE
            elif intensity_mae > 1.0:
                rec_confidence *= 0.9
        
        # Ensure confidence stays in valid range
        rec_confidence = max(0.0, min(1.0, rec_confidence))
        
        uncertainty_level = self.get_confidence_level(rec_confidence)
        uncertainty_info = self.uncertainty_levels[uncertainty_level]
        
        return {
            'confidence_score': round(rec_confidence, 3),
            'confidence_percent': round(rec_confidence * 100, 1),
            'uncertainty_level': uncertainty_level,
            'uncertainty_label': uncertainty_info['label'],
            'uncertainty_message': uncertainty_info['message'],
            'suggested_action': uncertainty_info['action']
        }
    
    def get_fallback_recommendation(self, emotion, intensity):
        """Get safe fallback recommendation when uncertain"""
        
        # Universal safe recommendations
        fallbacks = {
            'high_intensity': {
                'action': 'Take a break and breathe',
                'detail': 'When uncertain, focus on basic self-care',
                'duration': '5 minutes'
            },
            'low_intensity': {
                'action': 'Continue monitoring',
                'detail': 'Low intensity allows for observation',
                'duration': 'N/A'
            }
        }
        
        if intensity >= 4:
            return fallbacks['high_intensity']
        else:
            return fallbacks['low_intensity']
    
    def should_use_fallback(self, confidence_info):
        """Determine if fallback recommendation should be used"""
        
        # Use fallback when confidence is very low
        if confidence_info['confidence_score'] < 0.30:
            return True
        
        # Use fallback when uncertainty is high
        if confidence_info['uncertainty_level'] == 'high':
            return True
        
        return False
    
    def save_handler(self, filepath='models/confidence_handler.pkl'):
        """Save confidence handler to file"""
        
        Path(filepath).parent.mkdir(exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'thresholds': self.thresholds,
                'uncertainty_levels': self.uncertainty_levels,
                'uncertainty_factors': self.uncertainty_factors
            }, f)
        
        print(f"Saved: {filepath}")
    
    @staticmethod
    def load_handler(filepath='models/confidence_handler.pkl'):
        """Load confidence handler from file"""
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        handler = ConfidenceHandler()
        handler.thresholds = data['thresholds']
        handler.uncertainty_levels = data['uncertainty_levels']
        handler.uncertainty_factors = data['uncertainty_factors']
        
        return handler


# Test the confidence handler
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 5 - TASK 5.3: CONFIDENCE HANDLER")
    print("=" * 60)
    
    # Create handler
    handler = ConfidenceHandler()
    
    # Test different confidence scenarios
    print("\nConfidence Calculations for Different Scenarios:")
    print("-" * 60)
    
    test_cases = [
        # (model_proba, is_short, ambiguity_score, description)
        (np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.75]), False, 0.01, 'High confidence prediction'),
        (np.array([0.10, 0.15, 0.20, 0.25, 0.15, 0.15]), False, 0.02, 'Medium confidence prediction'),
        (np.array([0.20, 0.20, 0.20, 0.20, 0.10, 0.10]), True, 0.08, 'Low confidence + short text'),
        (np.array([0.30, 0.25, 0.20, 0.15, 0.05, 0.05]), True, 0.10, 'Very low confidence + ambiguous'),
    ]
    
    for model_proba, is_short, ambiguity, description in test_cases:
        text_features = {
            'is_short': is_short,
            'ambiguity_score': ambiguity
        }
        
        conf = handler.calculate_prediction_confidence(model_proba, text_features)
        
        print(f"\n{description}:")
        print(f"  Base Confidence: {conf['base_confidence']:.3f}")
        print(f"  Uncertainty Penalty: {conf['uncertainty_penalty']:.3f}")
        print(f"  Final Confidence: {conf['confidence_percent']:.1f}%")
        print(f"  Uncertainty Level: {conf['uncertainty_level']}")
    
    # Test recommendation confidence
    print("\n" + "=" * 60)
    print("Recommendation Confidence:")
    print("=" * 60)
    
    rec_test_cases = [
        (0.85, 0.7, 'High emotion confidence, low intensity MAE'),
        (0.60, 1.2, 'Medium emotion confidence, medium MAE'),
        (0.35, 1.8, 'Low emotion confidence, high MAE'),
    ]
    
    for emotion_conf, intensity_mae, description in rec_test_cases:
        rec_conf = handler.get_recommendation_confidence(emotion_conf, intensity_mae)
        
        print(f"\n{description}:")
        print(f"  Confidence: {rec_conf['confidence_percent']:.1f}%")
        print(f"  Uncertainty: {rec_conf['uncertainty_label']}")
        print(f"  Message: {rec_conf['uncertainty_message']}")
        print(f"  Suggested Action: {rec_conf['suggested_action']}")
    
    # Test fallback logic
    print("\n" + "=" * 60)
    print("Fallback Recommendation Test:")
    print("=" * 60)
    
    for conf_score in [0.85, 0.50, 0.25]:
        conf_info = {
            'confidence_score': conf_score,
            'uncertainty_level': handler.get_confidence_level(conf_score)
        }
        
        use_fallback = handler.should_use_fallback(conf_info)
        fallback = handler.get_fallback_recommendation('overwhelmed', 4)
        
        print(f"\nConfidence: {conf_score:.2f}")
        print(f"  Use Fallback: {use_fallback}")
        if use_fallback:
            print(f"  Fallback Action: {fallback['action']}")
    
    # Save handler
    print("\n" + "=" * 60)
    print("Saving Confidence Handler...")
    print("=" * 60)
    
    handler.save_handler()
    
    print("\nTASK 5.3 COMPLETE: Confidence Handler Created")