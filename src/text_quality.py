import pandas as pd 

class TextQuality:
    """Class for evaluating the quality of text data."""
    def __init__(self):
        self.ambiguity_words = {
            'maybe', 'perhaps', 'unsure', 'uncertain', 'kinda', 'sorta',
            'might', 'could', 'possibly', 'probably', 'guess', 'think'
        }

        self.coherence_words = {
            'and', 'but', 'because', 'so', 'therefore', 'however',
            'although', 'since', 'while', 'then', 'also', 'moreover'
        }

        self.emotion_words = {
            'happy', 'sad', 'angry', 'calm', 'anxious', 'excited',
            'tired', 'energetic', 'stressed', 'relaxed', 'focused',
            'overwhelmed', 'peaceful', 'frustrated', 'hopeful'
        }

    def quality_metrics(self, text):
        """Calculate quality metrics for the given text."""
        if pd.isna(text) or text == '' or str(text).strip() == '':
            return {
                'ambiguity_score': 1.0,
                'coherence_score': 0.0,
                'emotion_word_count': 0,
                'complexity_score': 0.0
            }
        
        text = str(text).lower()
        words = text.split()

        if len(words) == 0:
            return {
                'ambiguity_score': 1.0,
                'coherence_score': 0.0,
                'emotion_word_count': 0,
                'complexity_score': 0.0
            }
        
        ambiguity_count = sum(1 for word in words if word in self.ambiguity_words)
        coherence_count = sum(1 for word in words if word in self.coherence_words)
        emotion_count = sum(1 for word in words if word in self.emotion_words)
        ambiguity_score = ambiguity_count / len(words)
        coherence_score = coherence_count / len(words)
        unique_words = len(set(words))
        complexity_score = unique_words / len(words)

        return {
            'ambiguity_score': round(ambiguity_score, 3),
            'coherence_score': round(coherence_score, 3),
            'complexity_score': round(complexity_score, 3),
            'emotion_word_count': emotion_count
        }
    
    def process_columns(self, df, column_name = 'cleaned_text'):
        """Process the specified column in the DataFrame and add quality metrics."""
        print(f"Calculating quality metrics for {column_name}...")
        
        # Get metrics for each row
        metrics = df[column_name].apply(self.quality_metrics).apply(pd.Series)
        
        # Add to DataFrame
        df = pd.concat([df, metrics], axis=1)
        
        print(f"✅ Processed {len(df)} text entries")
        print(f"   Avg ambiguity: {metrics['ambiguity_score'].mean():.3f}")
        print(f"   Avg coherence: {metrics['coherence_score'].mean():.3f}")
        print(f"   Avg complexity: {metrics['complexity_score'].mean():.3f}")
        print(f"   Avg emotion words: {metrics['emotion_word_count'].mean():.2f}")
        
        return df

if __name__ == "__main__":
    df = pd.read_csv('data/train_data.csv')
    
    # First clean the text
    from text_cleaner import TextCleaner
    cleaner = TextCleaner()
    df = cleaner.process_columns(df)
    
    # Then analyze sentiment
    from sentiment_analyzer import SentimentAnalyzer
    sentiment = SentimentAnalyzer()
    df = sentiment.process_columns(df)
    
    # Then calculate quality metrics
    quality = TextQuality()
    df = quality.process_columns(df)
    
    print("\n📋 Sample quality metrics:")
    for i in range(3):
        print(f"\n{i+1}. Text: {df['cleaned_text'].iloc[i][:80]}...")
        print(f"   Ambiguity: {df['ambiguity_score'].iloc[i]:.3f}")
        print(f"   Coherence: {df['coherence_score'].iloc[i]:.3f}")
        print(f"   Complexity: {df['complexity_score'].iloc[i]:.3f}")
        print(f"   Emotion words: {df['emotion_word_count'].iloc[i]}")