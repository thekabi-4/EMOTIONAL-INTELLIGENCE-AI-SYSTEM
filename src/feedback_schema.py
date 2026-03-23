# src/feedback_schema.py
"""
Day 6 - Task 6.1: Feedback Schema Definition
Purpose: Define structure for collecting user feedback
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path

@dataclass
class FeedbackRecord:
    """Schema for a single feedback record"""
    
    # Required fields (NO defaults - must come first)
    feedback_id: str
    timestamp: str
    user_id: str
    emotion_predicted: str
    emotion_confidence: float
    intensity_predicted: int
    recommendation: str
    urgency_level: int
    confidence_level: str
    
    # Optional fields (WITH defaults - must come last)
    time_of_day: Optional[str] = None
    ambience_type: Optional[str] = None
    recommendation_detail: Optional[str] = None
    suggested_duration: Optional[str] = None
    user_followed: Optional[bool] = None
    outcome_rating: Optional[int] = None
    user_notes: Optional[str] = None
    model_version: str = "v1.0"
    feature_count: int = 47
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self, indent: bool = False) -> str:
        """
        Convert to JSON string
        
        Args:
            indent: If True, pretty-print with indentation (for display)
                   If False, single-line JSON (for JSONL files)
        """
        if indent:
            return json.dumps(self.to_dict(), indent=2)
        else:
            return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FeedbackRecord':
        """Create FeedbackRecord from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FeedbackRecord':
        """Create FeedbackRecord from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class FeedbackSchema:
    """Manage feedback schema and validation"""
    
    REQUIRED_FIELDS = [
        'feedback_id', 'timestamp', 'user_id',
        'emotion_predicted', 'emotion_confidence', 'intensity_predicted',
        'recommendation', 'urgency_level', 'confidence_level'
    ]
    
    OPTIONAL_FIELDS = [
        'time_of_day', 'ambience_type',
        'recommendation_detail', 'suggested_duration',
        'user_followed', 'outcome_rating', 'user_notes'
    ]
    
    VALID_EMOTIONS = ['calm', 'focused', 'mixed', 'neutral', 'overwhelmed', 'restless']
    VALID_URGENCY = [1, 2, 3, 4, 5]
    VALID_CONFIDENCE = ['low', 'medium', 'high']
    
    @staticmethod
    def validate(record: FeedbackRecord) -> tuple:
        """Validate a feedback record"""
        
        errors = []
        
        # Check required fields
        for field_name in FeedbackSchema.REQUIRED_FIELDS:
            if not hasattr(record, field_name) or getattr(record, field_name) is None:
                errors.append(f"Missing required field: {field_name}")
        
        # Validate emotion
        if record.emotion_predicted not in FeedbackSchema.VALID_EMOTIONS:
            errors.append(f"Invalid emotion: {record.emotion_predicted}")
        
        # Validate confidence
        if record.confidence_level not in FeedbackSchema.VALID_CONFIDENCE:
            errors.append(f"Invalid confidence level: {record.confidence_level}")
        
        # Validate urgency
        if record.urgency_level not in FeedbackSchema.VALID_URGENCY:
            errors.append(f"Invalid urgency level: {record.urgency_level}")
        
        # Validate confidence score range
        if not (0.0 <= record.emotion_confidence <= 1.0):
            errors.append(f"Confidence out of range: {record.emotion_confidence}")
        
        # Validate outcome rating if provided
        if record.outcome_rating is not None:
            if not (1 <= record.outcome_rating <= 10):
                errors.append(f"Outcome rating out of range: {record.outcome_rating}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def get_schema_documentation() -> dict:
        """Get schema documentation as JSON-serializable dict"""
        
        schema = {
            "required_fields": FeedbackSchema.REQUIRED_FIELDS,
            "optional_fields": FeedbackSchema.OPTIONAL_FIELDS,
            "field_types": {},
            "valid_values": {
                "emotions": FeedbackSchema.VALID_EMOTIONS,
                "urgency_levels": FeedbackSchema.VALID_URGENCY,
                "confidence_levels": FeedbackSchema.VALID_CONFIDENCE
            }
        }
        
        # Convert type annotations to strings
        for field_name, field_type in FeedbackRecord.__annotations__.items():
            schema["field_types"][field_name] = str(field_type)
        
        return schema


# Test the schema
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 6 - TASK 6.1: FEEDBACK SCHEMA")
    print("=" * 60)
    
    # Create a sample feedback record
    sample = FeedbackRecord(
        feedback_id="fb_001",
        timestamp=datetime.now().isoformat(),
        user_id="user_anonymous_001",
        emotion_predicted="overwhelmed",
        emotion_confidence=0.75,
        intensity_predicted=4,
        recommendation="breathing_exercise",
        urgency_level=5,
        confidence_level="low",
        time_of_day="morning",
        recommendation_detail="Try 4-7-8 breathing technique",
        suggested_duration="5 minutes",
        user_followed=True,
        outcome_rating=8,
        user_notes="Helped calm down quickly"
    )
    
    # Validate
    is_valid, errors = FeedbackSchema.validate(sample)
    
    print(f"\nSample Feedback Record:")
    print("-" * 60)
    print(sample.to_json(indent=True))
    
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"Errors: {errors}")
    
    # Save schema documentation
    Path('outputs').mkdir(exist_ok=True)
    
    schema_doc = FeedbackSchema.get_schema_documentation()
    with open('outputs/feedback_schema.json', 'w') as f:
        json.dump(schema_doc, f, indent=2)
    
    print(f"\nSchema documentation saved: outputs/feedback_schema.json")
    
    print("\n" + "=" * 60)
    print("TASK 6.1 COMPLETE: Feedback Schema Defined")
    print("=" * 60)