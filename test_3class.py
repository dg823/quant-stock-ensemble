"""
test_3class.py
----------------------
3개 클래스(매도, 중립, 매수) 분류 테스트 코드
제약: 원본 14개 피처만 사용
"""

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ── 경로 및 설정 ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Quant_Scaled_ML_Ready.csv")

FEATURES = [
    'retail_net_buy', 'foreign_net_buy', 'inst_net_buy',
    'close', 'volume', 'atr_14d', 'volatility_20d', 
    'disparity_20d', 'rsi_14d', 'vol_change_1d', 
    'macd_osc', 'bb_lower_break', 'vkospi', 'relative_strength'
]

# ── 데이터 로드 및 3클래스 라벨링 ───────────────────────────────────────────
def load_data(csv_path):
    print("[1/3] 데이터 로드 및 3클래스 라벨링 수행")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values(['name', 'date']).reset_index(drop=True)
    
    # 3일 뒤 가격 변화 (이미 스케일링된 데이터이므로 차이 사용)
    df['target_diff'] = df.groupby('name')['close'].transform(lambda x: x.shift(-3) - x)
    
    # 3개 클래스 정의 (균형 잡힌 분포를 위해 임계값 조정)
    # 0: 매도 (변화량 < -0.05)
    # 1: 중립 (-0.05 <= 변화량 < 0.05)
    # 2: 매수 (변화량 >= 0.05)
    def label_3class(diff):
        if diff < -0.05: return 0
        if diff < 0.05: return 1
        return 2

    df['label_3class'] = df['target_diff'].apply(label_3class)
    df = df.dropna(subset=FEATURES + ['target_diff'])
    return df

# ── 학습 및 평가 ─────────────────────────────────────────────────────────────
def run_test(df):
    print("[2/3] 데이터 분할 및 정규화 (8:2 비율)")
    X = df[FEATURES].values
    y = df['label_3class'].values
    
    # 시간 순서 유지를 위한 split (최근 20%를 테스트용으로 사용)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("[3/3] MLP 모델 학습 중 (3-Class Classification)")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=200, 
        random_state=42,
        early_stopping=True,
        learning_rate_init=0.001
    )
    clf.fit(X_train_s, y_train)
    
    y_pred = clf.predict(X_test_s)
    
    print("\n" + "="*60)
    print("                Classification Report (3-Class)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['매도(0)', '중립(1)', '매수(2)']))
    print("="*60)

if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        print(f"오류: {CSV_PATH} 파일을 찾을 수 없습니다.")
    else:
        data = load_data(CSV_PATH)
        run_test(data)
