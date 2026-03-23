# src/decision_pipeline.py
"""
Day 6 - Task 6.4: Decision Pipeline Integration
Purpose: Connect all components for end-to-end inference with feedback
"""

import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from text_pipeline import TextPipeline
from numerical_features import NumericalFeatureEngineer
from categorical_features import CategoricalFeatureEncoder
from feature_fusion import FeatureFusion
from recommendation_mapper import RecommendationMapper
from timing_engine import TimingEngine
from confidence_handler import ConfidenceHandler
from feedback_recorder import FeedbackRecorder


class DecisionPipeline:
    """End-to-end pipeline: journal entry → recommendation + feedback logging"""
    
    def __init__(self, model_dir: str = 'models', feedback_dir: str = 'data/feedback'):
        self.model_dir = Path(model_dir)
        self.feedback_dir = Path(feedback_dir)
        
        # Load models
        self._load_models()
        
        # Initialize ALL components
        self.text_pipeline = TextPipeline()
        self.num_engineer = NumericalFeatureEngineer()
        self.cat_encoder = CategoricalFeatureEncoder()
        self.feature_fusion = FeatureFusion()
        self.recommendation_mapper = RecommendationMapper()
        self.timing_engine = TimingEngine()
        self.confidence_handler = ConfidenceHandler()
        self.feedback_recorder = FeedbackRecorder(feedback_dir)
    
    def _load_models(self):
        """Load trained ML models and feature columns"""
        
        with open(self.model_dir / 'emotion_classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        
        with open(self.model_dir / 'intensity_regressor.pkl', 'rb') as f:
            self.regressor = pickle.load(f)
        
        with open(self.model_dir / 'emotion_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load trained feature columns (47 features)
        with open(self.model_dir / 'emotion_classifier_features.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        print(f"Loaded {len(self.feature_columns)} feature columns from model")
    
    def _extract_features(self, journal_text: str, metadata: Dict[str, Any]) -> dict:
        """Extract features from journal text and metadata"""
        
        import pandas as pd
        
        # Use default values that match training data categories
        defaults = {
            'duration_min': 30,
            'sleep_hours': 7,
            'energy_level': 3,
            'stress_level': 3,
            'previous_day_mood': 'neutral',
            'face_emotion_hint': 'neutral_face',
            'reflection_quality': 'clear',
            'time_of_day': 'morning',
            'ambience_type': 'cafe'
        }
        
        # Merge metadata with defaults
        full_metadata = {**defaults, **metadata}
        
        # Fix ambience_type if user provided invalid value
        valid_ambience = ['cafe', 'forest', 'mountain', 'ocean', 'rain']
        if full_metadata.get('ambience_type') not in valid_ambience:
            full_metadata['ambience_type'] = 'cafe'
        
        # Create temporary DataFrame for processing
        df = pd.DataFrame([{
            'journal_text': journal_text,
            **full_metadata
        }])
        
        # Run full feature pipeline
        df = self.text_pipeline.process(df)
        df = self.num_engineer.engineer_features(df)
        df = self.cat_encoder.encode_columns(df, auto_detect=True)
        
        X, _ = self.feature_fusion.fuse(df, include_targets=True)
        
        return X.iloc[0].to_dict()
    
    def _align_features(self, features: dict) -> 'pd.DataFrame':
        """Align features to match trained model's 47 feature columns exactly"""
        
        import pandas as pd
        import numpy as np
        
        # Create DataFrame with ALL 47 trained feature columns, initialized to 0
        X = pd.DataFrame([[0.0] * len(self.feature_columns)], columns=self.feature_columns)
        
        # Fill in actual feature values where column names match
        for col, val in features.items():
            if col in X.columns:
                try:
                    X[col] = pd.to_numeric([val], errors='coerce')[0]
                except:
                    X[col] = 0.0
        
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        
        # Ensure all columns are float
        for col in X.columns:
            X[col] = X[col].astype(float)
        
        return X
    
    def predict(self, journal_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate prediction and recommendation for a journal entry"""
        
        metadata = metadata or {}
        
        # Extract features
        features = self._extract_features(journal_text, metadata)
        
        # Align features to match trained model (exactly 47 columns)
        X = self._align_features(features)
        
        # Verify feature shape
        assert X.shape == (1, 47), f"Expected (1, 47) features, got {X.shape}"
        
        # Predict emotion
        emotion_proba = self.classifier.predict_proba(X)[0]
        emotion_encoded = self.classifier.predict(X)[0]
        emotion = self.label_encoder.inverse_transform([emotion_encoded])[0]
        emotion_confidence = float(emotion_proba.max())
        
        # Predict intensity
        intensity_raw = self.regressor.predict(X)[0]
        intensity = int(round(intensity_raw))
        intensity = max(1, min(5, intensity))
        
        # Get confidence info
        text_features = {
            'is_short': features.get('is_short', False),
            'ambiguity_score': features.get('ambiguity_score', 0)
        }
        confidence_info = self.confidence_handler.calculate_prediction_confidence(
            emotion_proba, text_features
        )
        rec_confidence = self.confidence_handler.get_recommendation_confidence(
            emotion_confidence, intensity_mae=None
        )
        
        # Get recommendation
        recommendation = self.recommendation_mapper.get_recommendation(emotion, intensity)
        
        # Get timing
        timing = self.timing_engine.get_timing(
            emotion, intensity,
            time_of_day=metadata.get('time_of_day'),
            category=recommendation.get('category')
        )
        
        # Build response
        response = {
            'input': {
                'journal_text': journal_text,
                'metadata': metadata
            },
            'predictions': {
                'emotion': emotion,
                'emotion_confidence': emotion_confidence,
                'intensity': intensity,
                'intensity_raw': float(intensity_raw)
            },
            'confidence': {
                'score': confidence_info['confidence_score'],
                'percent': confidence_info['confidence_percent'],
                'level': confidence_info['uncertainty_level'],
                'message': rec_confidence['uncertainty_message'],
                'suggested_action': rec_confidence['suggested_action']
            },
            'recommendation': {
                'action': recommendation['action'],
                'detail': recommendation['detail'],
                'category': recommendation['category'],
                'duration': recommendation['duration']
            },
            'timing': {
                'urgency_level': timing['urgency_level'],
                'urgency_label': timing['urgency_label'],
                'timeframe': timing['timeframe'],
                'recommendation': timing['timing_recommendation']
            },
            'feedback': {
                'feedback_id': None,
                'logged': False
            }
        }
        
        return response
    
    def log_feedback(
        self,
        response: Dict[str, Any],
        user_id: str,
        user_followed: bool,
        outcome_rating: Optional[int] = None,
        user_notes: Optional[str] = None
    ) -> str:
        """Log user feedback for a prediction"""
        
        predictions = response['predictions']
        recommendation = response['recommendation']
        timing = response['timing']
        confidence = response['confidence']
        metadata = response['input']['metadata']
        
        # Log feedback
        record = self.feedback_recorder.log_feedback(
            user_id=user_id,
            emotion_predicted=predictions['emotion'],
            emotion_confidence=predictions['emotion_confidence'],
            intensity_predicted=predictions['intensity'],
            recommendation=recommendation['action'],
            urgency_level=timing['urgency_level'],
            confidence_level=confidence['level'],
            time_of_day=metadata.get('time_of_day'),
            recommendation_detail=recommendation['detail'],
            suggested_duration=recommendation['duration'],
            user_followed=user_followed,
            outcome_rating=outcome_rating,
            user_notes=user_notes
        )
        
        response['feedback']['feedback_id'] = record.feedback_id
        response['feedback']['logged'] = True
        
        return record.feedback_id
    
    def get_user_history(self, user_id: str) -> list:
        """Get feedback history for a specific user"""
        return self.feedback_recorder.get_feedback_by_user(user_id)


# Test the integrated pipeline
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 6 - TASK 6.4: DECISION PIPELINE INTEGRATION")
    print("=" * 60)
    
    # Create pipeline
    pipeline = DecisionPipeline()
    
    # Test prediction
    print("\n" + "-" * 60)
    print("TEST PREDICTION")
    print("-" * 60)
    
    test_entry = "I feel overwhelmed with my workload today"
    test_metadata = {
        'time_of_day': 'morning',
        'ambience_type': 'cafe'
    }
    
    response = pipeline.predict(test_entry, test_metadata)
    
    print(f"\nInput: \"{test_entry}\"")
    print(f"\nPredictions:")
    print(f"  Emotion: {response['predictions']['emotion']} ({response['predictions']['emotion_confidence']:.0%} confidence)")
    print(f"  Intensity: {response['predictions']['intensity']}")
    
    print(f"\nConfidence:")
    print(f"  Level: {response['confidence']['level']}")
    print(f"  Message: {response['confidence']['message']}")
    
    print(f"\nRecommendation:")
    print(f"  Action: {response['recommendation']['action']}")
    print(f"  Detail: {response['recommendation']['detail']}")
    print(f"  Duration: {response['recommendation']['duration']}")
    
    print(f"\nTiming:")
    print(f"  Urgency: {response['timing']['urgency_label']} ({response['timing']['timeframe']})")
    print(f"  Advice: {response['timing']['recommendation']}")
    
    # Test feedback logging
    print("\n" + "-" * 60)
    print("TEST FEEDBACK LOGGING")
    print("-" * 60)
    
    feedback_id = pipeline.log_feedback(
        response=response,
        user_id="test_user_001",
        user_followed=True,
        outcome_rating=8,
        user_notes="The breathing exercise really helped calm me down"
    )
    
    print(f"\nFeedback logged: {feedback_id}")
    print(f"User: test_user_001")
    print(f"Followed: Yes")
    print(f"Rating: 8/10")
    
    print("\n" + "=" * 60)
    print("TASK 6.4 COMPLETE: Decision Pipeline Integrated")
    print("=" * 60)
    
    print(f"\nFull pipeline ready:")
    print(f"  1. pipeline.predict(journal_text, metadata) → recommendation")
    print(f"  2. pipeline.log_feedback(response, user_id, followed, rating) → feedback_id")
    print(f"  3. Feedback auto-saved to: {pipeline.feedback_recorder.filepath}")