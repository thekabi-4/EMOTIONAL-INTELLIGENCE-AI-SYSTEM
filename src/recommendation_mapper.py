# src/recommendation_mapper.py
"""
Day 5 - Task 5.1: Recommendation Mapper
Purpose: Map emotional states to specific actionable recommendations
"""

import pandas as pd
import pickle
from pathlib import Path

class RecommendationMapper:
    """Map emotional states to personalized recommendations"""
    
    def __init__(self):
        # Define recommendations for each emotional state
        self.recommendations = {
            'calm': {
                'action': 'Continue current activity',
                'detail': 'You are in a good state. Maintain this momentum.',
                'category': 'maintain',
                'duration': 'N/A'
            },
            'focused': {
                'action': 'Continue deep work session',
                'detail': 'You are focused. Consider using Pomodoro technique (25 min work, 5 min break).',
                'category': 'maintain',
                'duration': '25 minutes'
            },
            'mixed': {
                'action': 'Journal for 10 minutes',
                'detail': 'Your emotions are mixed. Writing can help clarify your thoughts.',
                'category': 'reflect',
                'duration': '10 minutes'
            },
            'neutral': {
                'action': 'Light stretching or walk',
                'detail': 'You are in a neutral state. Light movement can boost energy.',
                'category': 'activate',
                'duration': '5-10 minutes'
            },
            'overwhelmed': {
                'action': 'Breathing exercise',
                'detail': 'You seem overwhelmed. Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s.',
                'category': 'calm',
                'duration': '5 minutes'
            },
            'restless': {
                'action': 'Take a short walk',
                'detail': 'You seem restless. Physical movement can help release tension.',
                'category': 'activate',
                'duration': '10-15 minutes'
            }
        }
        
        # Define category descriptions
        self.category_descriptions = {
            'maintain': 'Continue current state',
            'reflect': 'Take time to reflect',
            'activate': 'Increase energy/movement',
            'calm': 'Reduce stress/calm down'
        }
    
    def get_recommendation(self, emotion, intensity=None):
        """Get recommendation for a given emotional state"""
        
        if emotion not in self.recommendations:
            return {
                'action': 'Take a break and reassess',
                'detail': 'Emotion not recognized. Consider taking a break.',
                'category': 'reflect',
                'duration': '5 minutes'
            }
        
        rec = self.recommendations[emotion].copy()
        
        # Adjust based on intensity if provided
        if intensity is not None:
            rec = self._adjust_for_intensity(rec, emotion, intensity)
        
        rec['emotion'] = emotion
        rec['intensity'] = intensity
        
        return rec
    
    def _adjust_for_intensity(self, rec, emotion, intensity):
        """Adjust recommendation based on intensity level"""
        
        # Low intensity (1-2): Gentle suggestions
        if intensity <= 2:
            rec['detail'] = f"(Low intensity) {rec['detail']}"
            rec['duration'] = '5 minutes'
        
        # High intensity (4-5): More urgent/stronger suggestions
        elif intensity >= 4:
            if emotion in ['overwhelmed', 'restless']:
                rec['action'] = f"IMMEDIATE: {rec['action']}"
                rec['detail'] = f"(High intensity) {rec['detail']} Consider longer session."
                rec['duration'] = '10-15 minutes'
            elif emotion in ['calm', 'focused']:
                rec['detail'] = f"(High intensity) {rec['detail']} Great state for important tasks!"
        
        return rec
    
    def get_all_recommendations(self):
        """Get all recommendations as a DataFrame"""
        
        rows = []
        for emotion, rec in self.recommendations.items():
            row = {
                'emotion': emotion,
                'action': rec['action'],
                'detail': rec['detail'],
                'category': rec['category'],
                'duration': rec['duration']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_mapping(self, filepath='models/recommendation_mapping.pkl'):
        """Save recommendation mapping to file"""
        
        Path(filepath).parent.mkdir(exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.recommendations, f)
        
        print(f"Saved: {filepath}")
    
    @staticmethod
    def load_mapping(filepath='models/recommendation_mapping.pkl'):
        """Load recommendation mapping from file"""
        
        with open(filepath, 'rb') as f:
            recommendations = pickle.load(f)
        
        mapper = RecommendationMapper()
        mapper.recommendations = recommendations
        
        return mapper


# Test the recommendation mapper
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 5 - TASK 5.1: RECOMMENDATION MAPPER")
    print("=" * 60)
    
    # Create mapper
    mapper = RecommendationMapper()
    
    # Display all recommendations
    print("\nAll Emotion Recommendations:")
    print("-" * 60)
    
    df_recs = mapper.get_all_recommendations()
    for _, row in df_recs.iterrows():
        print(f"\nEmotion: {row['emotion']}")
        print(f"  Action: {row['action']}")
        print(f"  Detail: {row['detail']}")
        print(f"  Category: {row['category']}")
        print(f"  Duration: {row['duration']}")
    
    # Test with different emotions and intensities
    print("\n" + "=" * 60)
    print("Sample Recommendations with Intensity")
    print("=" * 60)
    
    test_cases = [
        ('calm', 2),
        ('calm', 5),
        ('overwhelmed', 3),
        ('overwhelmed', 5),
        ('focused', 4),
        ('restless', 2),
        ('restless', 5)
    ]
    
    for emotion, intensity in test_cases:
        rec = mapper.get_recommendation(emotion, intensity)
        print(f"\n{emotion} (intensity {intensity}):")
        print(f"  Action: {rec['action']}")
        print(f"  Detail: {rec['detail']}")
        print(f"  Duration: {rec['duration']}")
    
    # Save mapping
    print("\n" + "=" * 60)
    print("Saving Recommendation Mapping...")
    print("=" * 60)
    
    mapper.save_mapping()
    
    print("\nTASK 5.1 COMPLETE: Recommendation Mapper Created")