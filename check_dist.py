import pandas as pd
import os

CSV_PATH = "Quant_Scaled_ML_Ready.csv"
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    # test_3class.py와 동일한 로직
    df['target_diff'] = df.groupby('name')['close'].transform(lambda x: x.shift(-3) - x)
    
    def label_3class(diff):
        if pd.isna(diff): return -1
        if diff < -0.05: return 0
        if diff < 0.05: return 1
        return 2

    df['label'] = df['target_diff'].apply(label_3class)
    print("Class Distribution:")
    print(df[df['label'] != -1]['label'].value_counts(normalize=True))
    print("\nCounts:")
    print(df[df['label'] != -1]['label'].value_counts())
else:
    print("CSV not found")
