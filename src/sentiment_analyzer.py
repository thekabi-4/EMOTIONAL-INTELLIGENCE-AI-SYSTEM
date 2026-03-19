# src/sentiment_analyzer.py
"""
Sentiment Analysis Module
Purpose: Analyze sentiment of journal text entries
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class SentimentAnalyzer:
    """Analyze sentiment of journal text entries"""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, text):
        if pd.isna(text) or str(text).strip() == "":
            return {
                "sentiment": "Neutral",
                "compound": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0
            }

        scores = self.analyzer.polarity_scores(str(text))

        if scores['compound'] >= 0.05:
            sentiment = "Positive"
        elif scores['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        scores['sentiment'] = sentiment
        return scores

    def process_columns(self, df, column_name='cleaned_text'):
        """Process entire column of text"""
        print(f"Analyzing sentiment for {column_name}...")
        sentiment_scores = df[column_name].apply(self.get_sentiment).apply(pd.Series)
        df = pd.concat([df, sentiment_scores], axis=1)
        print(f"Analyzed {len(df)} text entries")
        print(f"   Positive: {(df['sentiment'] == 'Positive').sum()}")
        print(f"   Neutral: {(df['sentiment'] == 'Neutral').sum()}")
        print(f"   Negative: {(df['sentiment'] == 'Negative').sum()}")
        print(f"   Average compound score: {df['compound'].mean():.3f}")
        return df


if __name__ == "__main__":
    df = pd.read_csv('data/train_data.csv')

    # First clean the text
    from text_cleaner import TextCleaner
    cleaner = TextCleaner()
    df = cleaner.process_columns(df)

    # Then analyze sentiment
    analyzer = SentimentAnalyzer()
    df = analyzer.process_columns(df)

    print("\nSample sentiment analysis:")
    for i in range(3):
        print(f"\n{i+1}. Text: {df['cleaned_text'].iloc[i][:80]}...")
        print(f"   Compound: {df['compound'].iloc[i]:.3f}")
        print(f"   Positive: {df['pos'].iloc[i]:.3f}")
        print(f"   Neutral: {df['neu'].iloc[i]:.3f}")
        print(f"   Negative: {df['neg'].iloc[i]:.3f}")
        print(f"   Sentiment: {df['sentiment'].iloc[i]}")
