import pandas as pd
import string

class TextCleaner:
    """clean and normalize journal text entries"""

    def __init__(self):
        pass

    def clean_text(self, text):
        if pd.isna(text) or str(text).strip() == "":
            return ""
        text = str(text).lower()
        text=' '.join(text.split())
        text=text.strip(string.punctuation + ' \t\n\r')
        return text
    
    def get_text_stats(self, text):
        """get statistics about the text"""
        if pd.isna(text) or str(text).strip() == "":
            return {
                "char_count": 0,
                "word_count": 0,
                "is_short": True
                }   
        text = str(text)
        words = text.split()
        return {
            "char_count": len(text),
            "word_count": len(words),
            "is_short": len(words) <= 5
        }
    
    def process_columns(self, df, column_name = 'journal_text'):
        """process entire column of text"""
        df['cleaned_text'] = df[column_name].apply(self.clean_text)
        stats = df[column_name].apply(self.get_text_stats).apply(pd.Series)
        df = pd.concat([df, stats], axis=1)
        print(f"✅ Processed {len(df)} text entries")
        print(f"   Short texts (≤5 words): {df['is_short'].sum()}")
        print(f"   Average word count: {df['word_count'].mean():.1f}")
        return df
        
if __name__ == "__main__":
    df = pd.read_csv('data/train_data.csv')
    cleaner = TextCleaner()
    df = cleaner.process_columns(df)
    
    print("\n📋 Sample cleaned texts:")
    for i in range(3):
        print(f"\n{i+1}. Original: {df['journal_text'].iloc[i][:80]}...")
        # ✅ Fixed: Use 'cleaned_text' to match column name
        print(f"   Cleaned: {df['cleaned_text'].iloc[i][:80]}...")
        print(f"   Words: {df['word_count'].iloc[i]}, Short: {df['is_short'].iloc[i]}")
        
