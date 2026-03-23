# src/feedback_recorder.py
"""
Day 6 - Task 6.2: Feedback Recorder
Purpose: Log user feedback to JSONL files for later analysis
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import uuid

from feedback_schema import FeedbackRecord, FeedbackSchema


class FeedbackRecorder:
    """Record and manage user feedback"""
    
    def __init__(self, feedback_dir: str = 'data/feedback'):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Get today's date for filename
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.filename = f"feedback_{self.today}.jsonl"
        self.filepath = self.feedback_dir / self.filename
    
    def _get_feedback_id(self) -> str:
        """Generate unique feedback ID"""
        return f"fb_{uuid.uuid4().hex[:8]}"
    
    def log_feedback(
        self,
        user_id: str,
        emotion_predicted: str,
        emotion_confidence: float,
        intensity_predicted: int,
        recommendation: str,
        urgency_level: int,
        confidence_level: str,
        time_of_day: Optional[str] = None,
        recommendation_detail: Optional[str] = None,
        suggested_duration: Optional[str] = None,
        user_followed: Optional[bool] = None,
        outcome_rating: Optional[int] = None,
        user_notes: Optional[str] = None
    ) -> FeedbackRecord:
        """
        Log a feedback record to file
        
        Returns:
            FeedbackRecord: The created record
        """
        
        # Create feedback record
        record = FeedbackRecord(
            feedback_id=self._get_feedback_id(),
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            emotion_predicted=emotion_predicted,
            emotion_confidence=emotion_confidence,
            intensity_predicted=intensity_predicted,
            recommendation=recommendation,
            urgency_level=urgency_level,
            confidence_level=confidence_level,
            time_of_day=time_of_day,
            recommendation_detail=recommendation_detail,
            suggested_duration=suggested_duration,
            user_followed=user_followed,
            outcome_rating=outcome_rating,
            user_notes=user_notes
        )
        
        # Validate
        is_valid, errors = FeedbackSchema.validate(record)
        if not is_valid:
            raise ValueError(f"Invalid feedback record: {errors}")
        
        # Append to JSONL file
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(record.to_json() + '\n')
        
        return record
    
    def get_feedback_count(self) -> int:
        """Get total number of feedback records for today"""
        if not self.filepath.exists():
            return 0
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def get_all_feedback(self) -> List[FeedbackRecord]:
        """Get all feedback records for today"""
        records = []
        
        if not self.filepath.exists():
            return records
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = FeedbackRecord.from_json(line)
                    records.append(record)
        
        return records
    
    def get_feedback_by_user(self, user_id: str) -> List[FeedbackRecord]:
        """Get all feedback records for a specific user"""
        all_feedback = self.get_all_feedback()
        return [r for r in all_feedback if r.user_id == user_id]
    
    def get_statistics(self) -> dict:
        """Get feedback statistics for today"""
        records = self.get_all_feedback()
        
        if not records:
            return {
                'total_records': 0,
                'avg_outcome_rating': None,
                'follow_rate': None,
                'emotion_distribution': {}
            }
        
        # Calculate statistics
        total = len(records)
        
        # Average outcome rating
        ratings = [r.outcome_rating for r in records if r.outcome_rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Follow rate
        followed = [r for r in records if r.user_followed is not None]
        follow_rate = sum(1 for r in followed if r.user_followed) / len(followed) if followed else None
        
        # Emotion distribution
        emotion_dist = {}
        for r in records:
            emotion = r.emotion_predicted
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        return {
            'total_records': total,
            'avg_outcome_rating': round(avg_rating, 2) if avg_rating else None,
            'follow_rate': round(follow_rate, 2) if follow_rate else None,
            'emotion_distribution': emotion_dist
        }


# Test the feedback recorder
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 6 - TASK 6.2: FEEDBACK RECORDER")
    print("=" * 60)
    
    # Create recorder
    recorder = FeedbackRecorder()
    
    print(f"\nFeedback directory: {recorder.feedback_dir}")
    print(f"Today's file: {recorder.filepath}")
    
    # Log sample feedback records
    print("\n" + "-" * 60)
    print("Logging Sample Feedback Records...")
    print("-" * 60)
    
    samples = [
        {
            'user_id': 'user_001',
            'emotion_predicted': 'overwhelmed',
            'emotion_confidence': 0.75,
            'intensity_predicted': 5,
            'recommendation': 'breathing_exercise',
            'urgency_level': 5,
            'confidence_level': 'low',
            'time_of_day': 'morning',
            'recommendation_detail': 'Try 4-7-8 breathing',
            'suggested_duration': '5 minutes',
            'user_followed': True,
            'outcome_rating': 8,
            'user_notes': 'Very helpful'
        },
        {
            'user_id': 'user_002',
            'emotion_predicted': 'restless',
            'emotion_confidence': 0.60,
            'intensity_predicted': 3,
            'recommendation': 'short_walk',
            'urgency_level': 3,
            'confidence_level': 'medium',
            'time_of_day': 'afternoon',
            'recommendation_detail': 'Walk around the block',
            'suggested_duration': '10 minutes',
            'user_followed': False,
            'outcome_rating': None,
            'user_notes': 'Too busy to walk'
        },
        {
            'user_id': 'user_001',
            'emotion_predicted': 'focused',
            'emotion_confidence': 0.85,
            'intensity_predicted': 4,
            'recommendation': 'deep_work',
            'urgency_level': 1,
            'confidence_level': 'low',
            'time_of_day': 'morning',
            'recommendation_detail': 'Continue current task',
            'suggested_duration': '25 minutes',
            'user_followed': True,
            'outcome_rating': 9,
            'user_notes': 'Great productivity'
        }
    ]
    
    for sample in samples:
        record = recorder.log_feedback(**sample)
        print(f"\nLogged: {record.feedback_id}")
        print(f"  User: {record.user_id}")
        print(f"  Emotion: {record.emotion_predicted}")
        print(f"  Recommendation: {record.recommendation}")
        print(f"  Followed: {record.user_followed}")
        print(f"  Rating: {record.outcome_rating}")
    
    # Get statistics
    print("\n" + "=" * 60)
    print("Feedback Statistics")
    print("=" * 60)
    
    stats = recorder.get_statistics()
    print(f"\nTotal records today: {stats['total_records']}")
    print(f"Average outcome rating: {stats['avg_outcome_rating']}")
    print(f"Follow rate: {stats['follow_rate']}")
    print(f"Emotion distribution: {stats['emotion_distribution']}")
    
    # Get records for specific user
    print("\n" + "=" * 60)
    print("User 001 Feedback History")
    print("=" * 60)
    
    user_records = recorder.get_feedback_by_user('user_001')
    for record in user_records:
        print(f"\n{record.feedback_id}:")
        print(f"  Emotion: {record.emotion_predicted}")
        print(f"  Recommendation: {record.recommendation}")
        print(f"  Rating: {record.outcome_rating}")
    
    print("\n" + "=" * 60)
    print("TASK 6.2 COMPLETE: Feedback Recorder Created")
    print("=" * 60)
    
    print(f"\nFeedback saved to: {recorder.filepath}")
    print("To view: code data/feedback/feedback_YYYY-MM-DD.jsonl")