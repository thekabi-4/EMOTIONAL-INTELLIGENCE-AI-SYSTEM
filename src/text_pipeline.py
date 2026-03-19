import pandas as pd
from text_cleaner import TextCleaner
from sentiment_analyzer import SentimentAnalyzer
from text_quality import TextQuality

class TextPipeline:
    """Complete text processing pipeline"""
    
    def __init__(self):
        # Initialize all components
        self.cleaner = TextCleaner()
        self.sentiment = SentimentAnalyzer()
        self.quality = TextQuality()
    
    def process(self, df, text_column='journal_text'):
        """Run full pipeline on DataFrame"""
        
        print(f"🔄 Starting text pipeline for column: {text_column}")
        
        # Step 1: Clean text
        print("  [1/3] Cleaning text...")
        df = self.cleaner.process_columns(df, text_column)
        
        # Step 2: Analyze sentiment
        print("  [2/3] Analyzing sentiment...")
        df = self.sentiment.process_columns(df, 'cleaned_text')
        
        # Step 3: Calculate quality metrics
        print("  [3/3] Calculating quality metrics...")
        df = self.quality.process_columns(df, 'cleaned_text')
        
        print(f"✅ Pipeline complete. Added {len(self.get_feature_columns())} text features")
        
        return df
    
    def get_feature_columns(self):
        """Return list of all text-derived feature columns"""
        return [
            'cleaned_text', 'char_count', 'word_count', 'is_short',
            'compound', 'pos', 'neu', 'neg', 'sentiment',
            'ambiguity_score', 'coherence_score', 'complexity_score', 'emotion_word_count'
        ]
    
    def process_single_text(self, text):
        """Process a single text string (for inference)"""
        
        # Clean
        cleaned = self.cleaner.clean_text(text)
        
        # Get stats
        stats = self.cleaner.get_text_stats(text)
        
        # Sentiment
        sentiment = self.sentiment.get_sentiment(cleaned)
        
        # Quality
        quality = self.quality.quality_metrics(cleaned)
        
        # Combine all
        return {
            'cleaned_text': cleaned,
            **stats,
            **sentiment,
            **quality
        }


# Test the pipeline
if __name__ == "__main__":
    df = pd.read_csv('data/train_data.csv')
    
    # Run full pipeline
    pipeline = TextPipeline()
    df = pipeline.process(df)
    
    print(f"\n📊 Final DataFrame shape: {df.shape}")
    print(f"📋 Text-derived columns: {pipeline.get_feature_columns()}")
    
    print("\n🔍 Sample row with all text features:")
    sample = df.iloc[0][pipeline.get_feature_columns()]
    for col, val in sample.items():
        print(f"   {col}: {val}")
    
    print("\n🧪 Test single text processing:")
    test_texts = [
        "I feel calm and focused today",
        "ok",
        "Maybe I should try harder but I'm not sure"
    ]
    
    for txt in test_texts:
        result = pipeline.process_single_text(txt)
        print(f"\n   Input: '{txt}'")
        print(f"   Cleaned: '{result['cleaned_text']}'")
        print(f"   Sentiment: {result['sentiment']} (compound: {result['compound']:.3f})")
        print(f"   Short: {result['is_short']}, Ambiguity: {result['ambiguity_score']:.3f}")