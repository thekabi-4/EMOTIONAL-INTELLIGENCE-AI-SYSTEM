# Data Validation Report

## Dataset Overview

- **Total Rows**: 1200 -**Total Columns**: 13
- **Missing Values**: id                      0
journal_text            0
ambience_type           0
duration_min            0
sleep_hours             7
energy_level            0
stress_level            0
time_of_day             0
previous_day_mood      15
face_emotion_hint     123
reflection_quality      0
emotional_state         0
intensity               0
dtype: int64
- **Emotional States**: 6
- **Intensity Range**: [1 to 5]
- **Data Types**:
id                      int64
journal_text              str
ambience_type             str
duration_min            int64
sleep_hours           float64
energy_level            int64
stress_level            int64
time_of_day               str
previous_day_mood         str
face_emotion_hint         str
reflection_quality        str
emotional_state           str
intensity               int64
dtype: object
- **Unique Previous Day Mood**: <ArrowStringArray>
['mixed', 'calm', 'overwhelmed', 'focused', nan, 'neutral', 'restless']
Length: 7, dtype: str
- **Unique Time of Day**: <ArrowStringArray>
['afternoon', 'evening', 'night', 'morning', 'early_morning']
Length: 5, dtype: str
- **Unique Ambience Type**: <ArrowStringArray>
['ocean', 'forest', 'mountain', 'rain', 'cafe']
Length: 5, dtype: str
- **Unique Face Emotion Hint**: <ArrowStringArray>
[   'calm_face',   'tired_face',   'happy_face',   'tense_face',
 'neutral_face',         'none',            nan]
Length: 7, dtype: str
- **Unique Emotional State**: <ArrowStringArray>
['focused', 'restless', 'calm', 'neutral', 'overwhelmed', 'mixed']
Length: 6, dtype: str
- **Unique Reflection Quality**: <ArrowStringArray>
['clear', 'vague', 'conflicted']
Length: 3, dtype: str
