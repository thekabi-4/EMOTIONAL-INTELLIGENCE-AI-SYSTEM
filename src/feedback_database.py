# src/feedback_database.py
"""
Day 6 - Task 6.3: Feedback Database
Purpose: Query and analyze collected feedback data
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd

from feedback_schema import FeedbackRecord


class FeedbackDatabase:
    """Query and analyze feedback data from JSONL files"""
    
    def __init__(self, feedback_dir: str = 'data/feedback'):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_all_files(self) -> List[FeedbackRecord]:
        """Load all feedback records from all JSONL files"""
        records = []
        
        # Find all JSONL files
        jsonl_files = list(self.feedback_dir.glob('feedback_*.jsonl'))
        
        for filepath in jsonl_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = FeedbackRecord.from_json(line)
                            records.append(record)
                        except Exception as e:
                            print(f"Warning: Failed to parse line in {filepath}: {e}")
        
        return records
    
    def get_all_records(self) -> List[FeedbackRecord]:
        """Get all feedback records"""
        return self._load_all_files()
    
    def get_records_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[FeedbackRecord]:
        """Get records within a date range (YYYY-MM-DD format)"""
        all_records = self._load_all_files()
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        filtered = []
        for record in all_records:
            record_date = datetime.fromisoformat(record.timestamp)
            if start <= record_date <= end:
                filtered.append(record)
        
        return filtered
    
    def get_records_by_emotion(self, emotion: str) -> List[FeedbackRecord]:
        """Get records for a specific emotion"""
        all_records = self._load_all_files()
        return [r for r in all_records if r.emotion_predicted == emotion]
    
    def get_records_by_recommendation(self, recommendation: str) -> List[FeedbackRecord]:
        """Get records for a specific recommendation"""
        all_records = self._load_all_files()
        return [r for r in all_records if r.recommendation == recommendation]
    
    def get_records_by_user(self, user_id: str) -> List[FeedbackRecord]:
        """Get all records for a specific user"""
        all_records = self._load_all_files()
        return [r for r in all_records if r.user_id == user_id]
    
    def analyze_recommendation_effectiveness(self) -> pd.DataFrame:
        """Analyze which recommendations work best (by avg outcome rating)"""
        all_records = self._load_all_files()
        
        # Group by recommendation
        rec_data = {}
        for record in all_records:
            rec = record.recommendation
            if rec not in rec_data:
                rec_data[rec] = {'ratings': [], 'followed': 0, 'total': 0}
            
            rec_data[rec]['total'] += 1
            if record.user_followed:
                rec_data[rec]['followed'] += 1
            if record.outcome_rating is not None:
                rec_data[rec]['ratings'].append(record.outcome_rating)
        
        # Calculate statistics
        rows = []
        for rec, data in rec_data.items():
            avg_rating = sum(data['ratings']) / len(data['ratings']) if data['ratings'] else None
            follow_rate = data['followed'] / data['total'] if data['total'] > 0 else None
            
            rows.append({
                'recommendation': rec,
                'avg_outcome_rating': round(avg_rating, 2) if avg_rating else None,
                'follow_rate': round(follow_rate, 2) if follow_rate else None,
                'total_uses': data['total'],
                'total_ratings': len(data['ratings'])
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('avg_outcome_rating', ascending=False)
        
        return df
    
    def analyze_emotion_outcomes(self) -> pd.DataFrame:
        """Analyze outcomes by emotion type"""
        all_records = self._load_all_files()
        
        # Group by emotion
        emotion_data = {}
        for record in all_records:
            emotion = record.emotion_predicted
            if emotion not in emotion_data:
                emotion_data[emotion] = {'ratings': [], 'followed': 0, 'total': 0}
            
            emotion_data[emotion]['total'] += 1
            if record.user_followed:
                emotion_data[emotion]['followed'] += 1
            if record.outcome_rating is not None:
                emotion_data[emotion]['ratings'].append(record.outcome_rating)
        
        # Calculate statistics
        rows = []
        for emotion, data in emotion_data.items():
            avg_rating = sum(data['ratings']) / len(data['ratings']) if data['ratings'] else None
            follow_rate = data['followed'] / data['total'] if data['total'] > 0 else None
            
            rows.append({
                'emotion': emotion,
                'avg_outcome_rating': round(avg_rating, 2) if avg_rating else None,
                'follow_rate': round(follow_rate, 2) if follow_rate else None,
                'total_records': data['total']
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('avg_outcome_rating', ascending=False)
        
        return df
    
    def get_daily_trends(self) -> pd.DataFrame:
        """Get daily feedback trends"""
        all_records = self._load_all_files()
        
        # Group by date
        daily_data = {}
        for record in all_records:
            date = record.timestamp[:10]  # YYYY-MM-DD
            if date not in daily_data:
                daily_data[date] = {'count': 0, 'ratings': []}
            
            daily_data[date]['count'] += 1
            if record.outcome_rating is not None:
                daily_data[date]['ratings'].append(record.outcome_rating)
        
        # Calculate statistics
        rows = []
        for date, data in sorted(daily_data.items()):
            avg_rating = sum(data['ratings']) / len(data['ratings']) if data['ratings'] else None
            
            rows.append({
                'date': date,
                'feedback_count': data['count'],
                'avg_outcome_rating': round(avg_rating, 2) if avg_rating else None
            })
        
        return pd.DataFrame(rows)
    
    def export_to_csv(self, output_path: str = 'outputs/feedback_analysis.csv'):
        """Export all feedback to CSV for external analysis"""
        all_records = self._load_all_files()
        
        if not all_records:
            print("No feedback records to export")
            return
        
        # Convert to list of dicts
        data = [record.to_dict() for record in all_records]
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        
        Path(output_path).parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"Exported {len(df)} records to: {output_path}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        all_records = self._load_all_files()
        
        if not all_records:
            return "No feedback records available"
        
        # Calculate overall statistics
        total = len(all_records)
        ratings = [r.outcome_rating for r in all_records if r.outcome_rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        followed = [r for r in all_records if r.user_followed is not None]
        follow_rate = sum(1 for r in followed if r.user_followed) / len(followed) if followed else 0
        
        # Get unique users
        unique_users = len(set(r.user_id for r in all_records))
        
        # Build report
        report = []
        report.append("=" * 60)
        report.append("FEEDBACK DATABASE SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total Feedback Records: {total}")
        report.append(f"Unique Users: {unique_users}")
        report.append(f"Average Outcome Rating: {avg_rating:.2f} / 10")
        report.append(f"Overall Follow Rate: {follow_rate:.2%}")
        report.append("")
        
        # Recommendation effectiveness
        report.append("-" * 60)
        report.append("RECOMMENDATION EFFECTIVENESS")
        report.append("-" * 60)
        rec_df = self.analyze_recommendation_effectiveness()
        if not rec_df.empty:
            for _, row in rec_df.iterrows():
                report.append(f"{row['recommendation']:25} | Rating: {row['avg_outcome_rating']:5} | Follow: {row['follow_rate']:.2%} | Uses: {row['total_uses']}")
        
        report.append("")
        
        # Emotion outcomes
        report.append("-" * 60)
        report.append("OUTCOMES BY EMOTION")
        report.append("-" * 60)
        emotion_df = self.analyze_emotion_outcomes()
        if not emotion_df.empty:
            for _, row in emotion_df.iterrows():
                report.append(f"{row['emotion']:15} | Rating: {row['avg_outcome_rating']:5} | Follow: {row['follow_rate']:.2%} | Records: {row['total_records']}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# Test the feedback database
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 6 - TASK 6.3: FEEDBACK DATABASE")
    print("=" * 60)
    
    # Create database
    db = FeedbackDatabase()
    
    # Get all records
    all_records = db.get_all_records()
    print(f"\nTotal records in database: {len(all_records)}")
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    report = db.generate_summary_report()
    print(report)
    
    # Export to CSV
    print("\n" + "=" * 60)
    print("EXPORTING DATA")
    print("=" * 60)
    
    db.export_to_csv()
    
    print("\n" + "=" * 60)
    print("TASK 6.3 COMPLETE: Feedback Database Created")
    print("=" * 60)