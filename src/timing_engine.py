# src/timing_engine.py
"""
Day 5 - Task 5.2: Timing Engine
Purpose: Determine urgency and best timing for recommendations
"""

import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

class TimingEngine:
    """Determine when to act based on emotion, intensity, and context"""
    
    def __init__(self):
        # Urgency levels
        self.urgency_levels = {
            1: {'label': 'Can Wait', 'color': 'green', 'timeframe': 'Within 24 hours'},
            2: {'label': 'Low Priority', 'color': 'blue', 'timeframe': 'Within 12 hours'},
            3: {'label': 'Medium Priority', 'color': 'yellow', 'timeframe': 'Within 4 hours'},
            4: {'label': 'High Priority', 'color': 'orange', 'timeframe': 'Within 1 hour'},
            5: {'label': 'URGENT', 'color': 'red', 'timeframe': 'ACT NOW'}
        }
        
        # Emotion urgency base scores (1-5)
        self.emotion_urgency = {
            'calm': 1,
            'focused': 1,
            'neutral': 2,
            'mixed': 3,
            'restless': 4,
            'overwhelmed': 5
        }
        
        # Time of day modifiers
        self.time_modifiers = {
            'early_morning': {'energy': 0.8, 'focus': 1.2},
            'morning': {'energy': 1.0, 'focus': 1.2},
            'afternoon': {'energy': 0.9, 'focus': 0.8},
            'evening': {'energy': 0.7, 'focus': 0.6},
            'night': {'energy': 0.5, 'focus': 0.4}
        }
        
        # Category to urgency mapping
        self.category_urgency = {
            'maintain': 1,
            'reflect': 3,
            'activate': 3,
            'calm': 5
        }
    
    def get_timing(self, emotion, intensity, time_of_day=None, category=None):
        """Calculate urgency and timing recommendation"""
        
        # Base urgency from emotion
        base_urgency = self.emotion_urgency.get(emotion, 3)
        
        # Adjust based on intensity (1-5 scale)
        intensity_factor = intensity / 3.0  # Normalize to ~0.33-1.67
        
        # Calculate final urgency score
        urgency_score = base_urgency * intensity_factor
        urgency_score = max(1, min(5, round(urgency_score)))
        
        # Get urgency level details
        urgency_info = self.urgency_levels[urgency_score]
        
        # Get timing recommendation
        timing_rec = self._get_timing_recommendation(emotion, urgency_score, time_of_day, category)
        
        return {
            'emotion': emotion,
            'intensity': intensity,
            'urgency_level': urgency_score,
            'urgency_label': urgency_info['label'],
            'urgency_color': urgency_info['color'],
            'timeframe': urgency_info['timeframe'],
            'timing_recommendation': timing_rec,
            'time_of_day': time_of_day
        }
    
    def _get_timing_recommendation(self, emotion, urgency, time_of_day, category):
        """Get specific timing recommendation based on context"""
        
        recommendations = {
            'overwhelmed': {
                5: 'Do breathing exercise immediately, before anything else',
                4: 'Take 5 minutes now to calm down',
                3: 'Find a quiet moment within the hour'
            },
            'restless': {
                5: 'Take a walk right now to release tension',
                4: 'Move your body within the next 30 minutes',
                3: 'Schedule a short walk soon'
            },
            'focused': {
                1: 'Continue your flow state, minimize interruptions',
                2: 'Stay in this state, consider Pomodoro technique'
            },
            'calm': {
                1: 'Maintain this state, good time for important tasks'
            },
            'mixed': {
                3: 'Journal when you have 10 uninterrupted minutes',
                4: 'Take time to reflect as soon as possible'
            },
            'neutral': {
                2: 'Light movement when convenient',
                3: 'Consider a short walk to boost energy'
            }
        }
        
        # Get emotion-specific recommendation
        if emotion in recommendations:
            rec = recommendations[emotion].get(urgency, 'Take action when convenient')
        else:
            rec = 'Take action when convenient'
        
        # Add time-of-day context if available
        if time_of_day and time_of_day in self.time_modifiers:
            mods = self.time_modifiers[time_of_day]
            if mods['focus'] < 0.7 and category == 'maintain':
                rec += ' (Note: Focus is typically lower at this time)'
            elif mods['energy'] < 0.7 and category == 'activate':
                rec += ' (Note: Energy is typically lower at this time)'
        
        return rec
    
    def get_best_time_to_act(self, time_of_day=None, category=None):
        """Suggest best time of day for different activity categories"""
        
        best_times = {
            'maintain': ['morning', 'early_morning'],
            'reflect': ['evening', 'night'],
            'activate': ['morning', 'afternoon'],
            'calm': ['afternoon', 'evening']
        }
        
        if category and category in best_times:
            return best_times[category]
        
        return ['morning', 'afternoon', 'evening']
    
    def save_engine(self, filepath='models/timing_engine.pkl'):
        """Save timing engine to file"""
        
        Path(filepath).parent.mkdir(exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'urgency_levels': self.urgency_levels,
                'emotion_urgency': self.emotion_urgency,
                'time_modifiers': self.time_modifiers,
                'category_urgency': self.category_urgency
            }, f)
        
        print(f"Saved: {filepath}")
    
    @staticmethod
    def load_engine(filepath='models/timing_engine.pkl'):
        """Load timing engine from file"""
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        engine = TimingEngine()
        engine.urgency_levels = data['urgency_levels']
        engine.emotion_urgency = data['emotion_urgency']
        engine.time_modifiers = data['time_modifiers']
        engine.category_urgency = data['category_urgency']
        
        return engine


# Test the timing engine
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 5 - TASK 5.2: TIMING ENGINE")
    print("=" * 60)
    
    # Create engine
    engine = TimingEngine()
    
    # Test different scenarios
    print("\nTiming Recommendations for Different Scenarios:")
    print("-" * 60)
    
    test_cases = [
        ('overwhelmed', 5, 'morning', 'calm'),
        ('overwhelmed', 3, 'afternoon', 'calm'),
        ('restless', 5, 'evening', 'activate'),
        ('restless', 2, 'morning', 'activate'),
        ('focused', 4, 'morning', 'maintain'),
        ('focused', 2, 'afternoon', 'maintain'),
        ('calm', 2, 'morning', 'maintain'),
        ('calm', 5, 'morning', 'maintain'),
        ('mixed', 3, 'evening', 'reflect'),
        ('neutral', 2, 'afternoon', 'activate')
    ]
    
    for emotion, intensity, time_of_day, category in test_cases:
        timing = engine.get_timing(emotion, intensity, time_of_day, category)
        print(f"\n{emotion} (intensity {intensity}) - {time_of_day}:")
        print(f"  Urgency: {timing['urgency_level']} ({timing['urgency_label']})")
        print(f"  Timeframe: {timing['timeframe']}")
        print(f"  Recommendation: {timing['timing_recommendation']}")
    
    # Save engine
    print("\n" + "=" * 60)
    print("Saving Timing Engine...")
    print("=" * 60)
    
    engine.save_engine()
    
    print("\nTASK 5.2 COMPLETE: Timing Engine Created")